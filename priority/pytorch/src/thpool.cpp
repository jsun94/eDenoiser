#define _POSIX_C_SOURCE 200809L
//#define _GNU_SOURCE
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include "thpool.h"
#include "net.h"
//#include "utils.h"
#include <sched.h>

#ifdef THPOOL_DEBUG
#define THPOOL_DEBUG 1
#else
#define THPOOL_DEBUG 0
#endif

#if !defined(DISABLE_PRINT) || defined(THPOOL_DEBUG)
#define err(str) fprintf(stderr, str)
#else
#define err(str)
#endif

#define RT_THREAD 1
#define GAP 2
#define MPANIC(x) ; assert(x != NULL) //조건에 맞지않으면 중단



static volatile int threads_keepalive;
static volatile int threads_on_hold;


/* ========================== PROTOTYPES ============================ */


static int  thread_init(thpool_* thpool_p, struct thread** thread_p, int id);
static void* thread_do(struct thread* thread_p);
static void  thread_hold(int sig_id);
static void  thread_destroy(struct thread* thread_p);

static int   jobqueue_init(jobqueue* jobqueue_p);
static void  jobqueue_clear(jobqueue* jobqueue_p);
static void  jobqueue_push(jobqueue* jobqueue_p, struct job* newjob_p);
static struct job* jobqueue_pull(jobqueue* jobqueue_p);
static void  jobqueue_destroy(jobqueue* jobqueue_p);

static void  bsem_init(struct bsem *bsem_p, int value);
static void  bsem_reset(struct bsem *bsem_p);
static void  bsem_post(struct bsem *bsem_p);
static void  bsem_post_all(struct bsem *bsem_p);
static void  bsem_wait(struct bsem *bsem_p);

static void insert_job(Priqueue *priqueue,struct job *newjob);
static job* pop_job(Priqueue *priqueue);
static void swap_job(Priqueue *priqueue,unsigned int a, unsigned int b);


/* ========================== THREADPOOL ============================ */


/* Initialise thread pool */
#if FIFOQ
struct thpool_* thpool_init(int num_threads){
#else
struct thpool_* thpool_init(int num_threads, int n_all){
#endif
	
	threads_on_hold   = 0;
	threads_keepalive = 1;

	if (num_threads < 0){
		num_threads = 0;
	}

	/* Make new thread pool */
	thpool_ *thpool_p;
	thpool_p = (struct thpool_*)malloc(sizeof(struct thpool_));
	if (thpool_p == NULL){
		err("thpool_init(): Could not allocate memory for thread pool\n");
		return NULL;
	}

	
	thpool_p->num_threads_alive   = 0;
	thpool_p->num_threads_working = 0;
	#if FIFOQ

	#else
	thpool_p->thread_length = num_threads;
	#endif
	
	#if FIFOQ
	/* Initialise the job queue */
	if (jobqueue_init(&thpool_p->jobqueue) == -1){
		err("thpool_init(): Could not allocate memory for job queue\n");
		free(thpool_p);
		return NULL;
	}
	
	#else
	/* Initialise the job queue */
	if ((thpool_p->priqueue = priqueue_init(n_all))==NULL)
	{
		err("thpool_init(): Could not allocate memory for job queue\n");
		free(thpool_p);
		return NULL;
	}
	#endif

	/* Make threads in pool */
	thpool_p->threads = (struct thread**)malloc(num_threads * sizeof(struct thread *));
	if (thpool_p->threads == NULL){
		err("thpool_init(): Could not allocate memory for threads\n");
		#if FIFOQ
		jobqueue_destroy(&thpool_p->jobqueue);
		#else
		priqueue_free(thpool_p->priqueue);
		#endif
		
		free(thpool_p);
		return NULL;
	}

	pthread_mutex_init(&(thpool_p->thcount_lock), NULL);
	pthread_cond_init(&thpool_p->threads_all_idle, NULL);

	/* Thread init */
	int n;
	cpu_set_t cpuset;

	for (n=0; n<num_threads; n++){
		thread_init(thpool_p, &thpool_p->threads[n], n);
		#if FIFOQ
		if(n == (num_threads-1)){
			thpool_p->threads[n]->flag = 1;
		}
		else{
			thpool_p->threads[n]->flag = 0;
		}
		
 
		/* kmsjames 2020 0215 bug fix for pinning each thread on a specified CPU */
		#else
		CPU_ZERO(&cpuset);
		CPU_SET(n, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
		pthread_setaffinity_np(thpool_p->threads[n]->pthread, sizeof(cpu_set_t), &cpuset);
		#endif

#if THPOOL_DEBUG
		printf("THPOOL_DEBUG: Created thread %d in pool \n", n);
#endif
	}

	/* Wait for threads to initialize */

	while (thpool_p->num_threads_alive != num_threads)
	{
	}

	return thpool_p;
}


/* Add work to the thread pool */
int thpool_add_work(thpool_* thpool_p, void (*function_p)(void*), void *arg_p){
	job* newjob;

	newjob=(struct job*)malloc(sizeof(struct job)); 
	if (newjob==NULL){
		err("thpool_add_work(): Could not allocate memory for new job\n");
		return -1;
	}

	/* add function and argument */
	newjob->function=function_p;
	newjob->arg=arg_p;

	#if FIFOQ

	#else

	newjob->priority = ((th_arg*)arg_p)->arg->net->priority;	// priority Queue 적용
	//std::cout << "new job priority" << newjob->priority << std::endl;
	#endif

	
	/* add job to queue */
	#if FIFOQ
	jobqueue_push(&thpool_p->jobqueue, newjob);
	#else	
	priqueue_insert(thpool_p->priqueue, newjob);	// priority Queue 적용
	#endif


	return 0;
}


/* Wait until all jobs have finished */
void thpool_wait(thpool_* thpool_p){
	pthread_mutex_lock(&thpool_p->thcount_lock);
	while (thpool_p->jobqueue.len || thpool_p->num_threads_working) {
		pthread_cond_wait(&thpool_p->threads_all_idle, &thpool_p->thcount_lock);
	}
	pthread_mutex_unlock(&thpool_p->thcount_lock);
}


/* Destroy the threadpool */
void thpool_destroy(thpool_* thpool_p){
	/* No need to destory if it's NULL */
	if (thpool_p == NULL) return ;

	volatile int threads_total = thpool_p->num_threads_alive;

	/* End each thread 's infinite loop */
	threads_keepalive = 0;

	/* Give one second to kill idle threads*/
	double TIMEOUT = 1.0;
	time_t start, end;
	double tpassed = 0.0;
	time (&start);
	while (tpassed < TIMEOUT && thpool_p->num_threads_alive){
		bsem_post_all(thpool_p->jobqueue.has_jobs);
		time (&end);
		tpassed = difftime(end,start);
	}

	/* Poll remaining threads */
	while (thpool_p->num_threads_alive){
		fprintf(stderr,"xxxxxxxxxxxxxxx\n");
		bsem_post_all(thpool_p->jobqueue.has_jobs);
		sleep(1);
	}

	/* Job queue cleanup */
	jobqueue_destroy(&thpool_p->jobqueue);
	/* Deallocs */
	int n;
	for (n=0; n < threads_total; n++){
		thread_destroy(thpool_p->threads[n]);
	}
	free(thpool_p->threads);
	free(thpool_p);
}


/* Pause all threads in threadpool */
void thpool_pause(thpool_* thpool_p) {
	int n;
	for (n=0; n < thpool_p->num_threads_alive; n++){
		pthread_kill(thpool_p->threads[n]->pthread, SIGUSR1);
	}
}


/* Resume all threads in threadpool */
void thpool_resume(thpool_* thpool_p) {
    // resuming a single threadpool hasn't been
    // implemented yet, meanwhile this supresses
    // the warnings
    (void)thpool_p;

	threads_on_hold = 0;
}


int thpool_num_threads_working(thpool_* thpool_p){
	return thpool_p->num_threads_working;
}





/* ============================ THREAD ============================== */


/* Initialize a thread in the thread pool
 *
 * @param thread        address to the pointer of the thread to be created
 * @param id            id to be given to the thread
 * @return 0 on success, -1 otherwise.
 */
static int thread_init (thpool_* thpool_p, struct thread** thread_p, int id){

#if FIFOQ

#else
        pthread_attr_t attr;
        struct sched_param param;

        pthread_attr_init(&attr);
        pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
        param.sched_priority = 99;
        pthread_attr_setschedparam(&attr, &param);
#endif

	*thread_p = (struct thread*)malloc(sizeof(struct thread));
	if (*thread_p == NULL){
		err("thread_init(): Could not allocate memory for thread\n");
		return -1;
	}

	(*thread_p)->thpool_p = thpool_p;
	(*thread_p)->id       = id;

#if FIFOQ
		pthread_create(&(*thread_p)->pthread, NULL, (void* (*)(void*))thread_do, (*thread_p));
#else
		pthread_create(&(*thread_p)->pthread, &attr, (void* (*)(void*))thread_do, (*thread_p));
#endif

	pthread_detach((*thread_p)->pthread);
	return 0;
}


/* Sets the calling thread on hold */
static void thread_hold(int sig_id) {
    (void)sig_id;
	threads_on_hold = 1;
	while (threads_on_hold){
		sleep(1);
	}
}


/* What each thread is doing
*
* In principle this is an endless loop. The only time this loop gets interuppted is once
* thpool_destroy() is invoked or the program exits.
*
* @param  thread        thread that will run this function
* @return nothing
*/
static void *thread_do(struct thread *thread_p)
{

	/* Set thread name for profiling and debuging */
	char thread_name[128] = {0};
	sprintf(thread_name, "thread-pool-%d", thread_p->id);

#if defined(__linux__)
	/* Use prctl instead to prevent using _GNU_SOURCE flag and implicit declaration */
	prctl(PR_SET_NAME, thread_name);
#elif defined(__APPLE__) && defined(__MACH__)
	pthread_setname_np(thread_name);
#else
	err("thread_do(): pthread_setname_np is not supported on this system");
#endif

	/* Assure all threads have been created before starting serving */
	thpool_ *thpool_p = thread_p->thpool_p;

	/* Register signal handler */
	struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	act.sa_handler = thread_hold;
	if (sigaction(SIGUSR1, &act, NULL) == -1)
	{
		err("thread_do(): cannot handle SIGUSR1");
	}

	/* Mark thread as alive (initialized) */
	pthread_mutex_lock(&thpool_p->thcount_lock);
	thpool_p->num_threads_alive += 1;
	pthread_mutex_unlock(&thpool_p->thcount_lock);

	while (threads_keepalive)
	{
		#if FIFOQ
		bsem_wait(thpool_p->jobqueue.has_jobs);
		#else
		bsem_wait(thpool_p->priqueue->hasjobs);
		#endif

		if (threads_keepalive)
		{

			pthread_mutex_lock(&thpool_p->thcount_lock);
			thpool_p->num_threads_working++;
			pthread_mutex_unlock(&thpool_p->thcount_lock);

			/* Read job from queue and execute it*/
			void (*func_buff)(void *); // buuffer = 데이터 임시 저장
			void *arg_buff;
			#if FIFOQ
			job *job_p = jobqueue_pull(&thpool_p->jobqueue);
			#else
			job *job_p = priqueue_pop(thpool_p->priqueue);
			#endif
			//std::cout << "poped job`s priority" << job_p->priority << std::endl;
			if (job_p)
			{
				func_buff = job_p->function;
				arg_buff = job_p->arg;
				#if FIFOQ
				((th_arg*)arg_buff)->flag = 0;  //thread_p->flag;
				#else

				#endif
				//thread_p->exe_time = job_p->exe_time;
				//fprintf(timing, "%d,%lf\n", ((netlayer*)arg_buff)->net.index_n, what_time_is_it_now());
				func_buff(arg_buff);
				free(job_p);
			}

			pthread_mutex_lock(&thpool_p->thcount_lock);
			thpool_p->num_threads_working--;
			if (!thpool_p->num_threads_working)
			{
				pthread_cond_signal(&thpool_p->threads_all_idle);
			}
			pthread_mutex_unlock(&thpool_p->thcount_lock);
		}
	}
	pthread_mutex_lock(&thpool_p->thcount_lock);
	thpool_p->num_threads_alive--;
	pthread_mutex_unlock(&thpool_p->thcount_lock);;
	return NULL;
}


/* Frees a thread  */
static void thread_destroy (thread* thread_p){
	free(thread_p);
}





/* ============================ JOB QUEUE =========================== */


/* Initialize queue */
static int jobqueue_init(jobqueue* jobqueue_p){
	jobqueue_p->len = 0;
	jobqueue_p->front = NULL;
	jobqueue_p->rear  = NULL;

	jobqueue_p->has_jobs = (struct bsem*)malloc(sizeof(struct bsem));
	if (jobqueue_p->has_jobs == NULL){
		return -1;
	}

	pthread_mutex_init(&(jobqueue_p->rwmutex), NULL);
	bsem_init(jobqueue_p->has_jobs, 0);

	return 0;
}


/* Clear the queue */
static void jobqueue_clear(jobqueue* jobqueue_p){

	while(jobqueue_p->len){
		free(jobqueue_pull(jobqueue_p));
	}

	jobqueue_p->front = NULL;
	jobqueue_p->rear  = NULL;
	bsem_reset(jobqueue_p->has_jobs);
	jobqueue_p->len = 0;

}


/* Add (allocated) job to queue
 */
static void jobqueue_push(jobqueue* jobqueue_p, struct job* newjob){

	//std::cout << "jobQ push" << std::endl;
	pthread_mutex_lock(&jobqueue_p->rwmutex);
	newjob->prev = NULL;

	switch(jobqueue_p->len){

		case 0:  /* if no jobs in queue */
					jobqueue_p->front = newjob;
					jobqueue_p->rear  = newjob;
					break;

		default: /* if jobs in queue */
					jobqueue_p->rear->prev = newjob;
					jobqueue_p->rear = newjob;

	}
	jobqueue_p->len++;

	bsem_post(jobqueue_p->has_jobs);
	pthread_mutex_unlock(&jobqueue_p->rwmutex);
}


/* Get first job from queue(removes it from queue)
<<<<<<< HEAD
 *
 * Notice: Caller MUST hold a mutex
=======
>>>>>>> da2c0fe45e43ce0937f272c8cd2704bdc0afb490
 */
static struct job* jobqueue_pull(jobqueue* jobqueue_p){

	pthread_mutex_lock(&jobqueue_p->rwmutex);
	job* job_p = jobqueue_p->front;

	switch(jobqueue_p->len){

		case 0:  /* if no jobs in queue */
		  			break;

		case 1:  /* if one job in queue */
					jobqueue_p->front = NULL;
					jobqueue_p->rear  = NULL;
					jobqueue_p->len = 0;
					break;

		default: /* if >1 jobs in queue */
					jobqueue_p->front = job_p->prev;
					jobqueue_p->len--;
					/* more than one job in queue -> post it */
					bsem_post(jobqueue_p->has_jobs);

	}

	pthread_mutex_unlock(&jobqueue_p->rwmutex);
	return job_p;
}


/* Free all queue resources back to the system */
static void jobqueue_destroy(jobqueue* jobqueue_p){
	jobqueue_clear(jobqueue_p);
	free(jobqueue_p->has_jobs);
}


/* ======================== PRIORITY QUEUE ========================= */

Priqueue* priqueue_init(int init_length){
  unsigned int mutex_status;
  Priqueue *priqueue = (Priqueue *) malloc(sizeof(Priqueue)) MPANIC(priqueue);  
  const size_t qsize = init_length * sizeof(*priqueue->array);
  priqueue->hasjobs = (bsem *)malloc(sizeof(struct bsem));
  if(priqueue->hasjobs==NULL){
    return NULL;
  }

  mutex_status = pthread_mutex_init(&(priqueue->lock), NULL);
  bsem_init(priqueue->hasjobs, 0);
  if (mutex_status != 0) goto error;
  
  priqueue->head = NULL;
  priqueue->heap_size = init_length; //need?
  priqueue->occupied = 0;
  priqueue->current = 1;
  priqueue->array = (job**)malloc(qsize) MPANIC(priqueue->array);

  memset(priqueue->array, 0x00, qsize);
  
  return priqueue;
  
 error:
  free(priqueue);
  printf("stop\n");

  return NULL;
}

static MHEAP_API MHEAPSTATUS realloc_heap(Priqueue *priqueue){

  if (priqueue->occupied >= priqueue->heap_size){
    const size_t arrsize = sizeof(*priqueue->array);
    
    void **resized_queue;
    resized_queue = (void**)realloc(priqueue->array, (2 * priqueue->heap_size) * arrsize);
    if (resized_queue != NULL){
      priqueue->heap_size *= 2;
      priqueue->array = (job**) resized_queue;
      memset( (priqueue->array + priqueue->occupied + 1) , 0x00, (priqueue->heap_size / GAP) * arrsize );
      return MHEAP_OK;
    } else return MHEAP_REALLOCERROR;
  }

  return MHEAP_NOREALLOC;
}

 
void priqueue_insert(Priqueue *priqueue, struct job *newjob){
  pthread_mutex_lock(&(priqueue->lock));
  insert_job(priqueue,newjob);  
  bsem_post(priqueue->hasjobs);
  pthread_mutex_unlock(&(priqueue->lock));
}

static void insert_job(Priqueue *priqueue, struct job* newjob){
  if (priqueue->current == 1 || priqueue->array[1] == NULL){
    priqueue->head = newjob;
    priqueue->array[1] = newjob;
    priqueue->array[1]->index = priqueue->current;
    priqueue->occupied++;
    priqueue->current++;
    
    return;
  }

  if(priqueue->occupied >= priqueue->heap_size) {
    unsigned int realloc_status = realloc_heap(priqueue);
    assert(realloc_status == MHEAP_OK);
  }
  
  if(priqueue->occupied <= priqueue->heap_size){
    newjob->index = priqueue->current;
    priqueue->array[priqueue->current] = newjob;

    int parent = (priqueue->current / GAP);

    if (priqueue->array[parent]->priority < newjob->priority){ //확인 필요
      priqueue->occupied++;
      priqueue->current++;
      int depth = priqueue->current / GAP;
      int traverse = newjob->index;
	  
	  while(depth >= 1){
		  if (traverse == 1) break;
		  unsigned int parent = (traverse / GAP);
		  
		  if(priqueue->array[parent]->priority < priqueue->array[traverse]->priority){
			swap_job(priqueue, parent , traverse);
        	traverse = priqueue->array[parent]->index;
		  }
		  depth --;
      }
    priqueue->head = priqueue->array[1];
    }else{
    	priqueue->occupied++;
    	priqueue->current++;
    }
  }
}

void swap_job(Priqueue *priqueue, unsigned int parent, unsigned int child){
  job *tmp = priqueue->array[parent];

  priqueue->array[parent] = priqueue->array[child];
  priqueue->array[parent]->index = tmp->index;

  priqueue->array[child] = tmp;
  priqueue->array[child]->index = child;
  
}

MHEAP_API job *priqueue_pop(Priqueue *priqueue){
  job *job_p = NULL;
  
  pthread_mutex_lock(&(priqueue->lock));
  job_p = pop_job(priqueue);
  pthread_mutex_unlock(&(priqueue->lock));

  return job_p;
}

static job *pop_job(Priqueue *priqueue){
  job *job_p = NULL;
  unsigned int i;
  unsigned int depth;

  if (priqueue->current == 1) return job_p;
  
  else if (priqueue->current >= 2 ){
    job_p = priqueue->array[1];
    priqueue->array[1] = priqueue->array[priqueue->current - 1];
    priqueue->current -= 1;
    priqueue->occupied -= 1;
    
    depth = (priqueue->current -1) / 2;

    for(i = 1; i<=depth; i++){
      
      if (priqueue->array[i]->priority < priqueue->array[i * GAP]->priority || priqueue->array[i]->priority < priqueue->array[(i * GAP)+1]->priority){
        unsigned int biggest = priqueue->array[i * GAP]->priority > priqueue->array[(i * GAP)+1]->priority ?
	      priqueue->array[(i * GAP)]->index  :
	      priqueue->array[(i * GAP)+1]->index;
        swap_job(priqueue,i,biggest);
      }
    }if(priqueue->current != 1)
    	bsem_post(priqueue->hasjobs);
  }

  return job_p;
}

MHEAP_API void priqueue_free(Priqueue *priqueue){  
  bsem_reset(priqueue->hasjobs);
  if (priqueue->current >= 2 ) {
    unsigned int i;
    for (i = 1; i <= priqueue->current; i++) priqueue_job_free(priqueue,priqueue->array[i]);
  }

  free(priqueue->head);
  free(*priqueue->array);
  free(priqueue->array);
  free(priqueue);
}

MHEAP_API void priqueue_job_free(Priqueue *priqueue,job *job_p){
  //if (node != NULL) free(node->data->data);
  free(job_p);  
}


/* ======================== SYNCHRONISATION ========================= */


/* Init semaphore to 1 or 0 */
static void bsem_init(bsem *bsem_p, int value) {
	if (value < 0 || value > 1) {
		err("bsem_init(): Binary semaphore can take only values 1 or 0");
		exit(1);
	}
	pthread_mutex_init(&(bsem_p->mutex), NULL);
	pthread_cond_init(&(bsem_p->cond), NULL);
	bsem_p->v = value;
}


/* Reset semaphore to 0 */
static void bsem_reset(bsem *bsem_p) {
	bsem_init(bsem_p, 0);
}


/* Post to at least one thread */
static void bsem_post(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_signal(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}


/* Post to all threads */
static void bsem_post_all(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_broadcast(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}


/* Wait on semaphore until semaphore has value 0 */
static void bsem_wait(bsem* bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	while (bsem_p->v != 1) {
		pthread_cond_wait(&bsem_p->cond, &bsem_p->mutex);
	}
	bsem_p->v = 0;
	pthread_mutex_unlock(&bsem_p->mutex);
}
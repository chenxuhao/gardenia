#ifndef __LB_HPP__
#define __LB_HPP__

#include <omp.h>
#include <deque>
#include <string>
#include <vector>
#include <graph_types.hpp>

class lb {
protected:
	typedef enum { RT_WORK_REQUEST = 0, RT_WORK_RESPONSE = 1, GLOBAL_WORK_REQUEST = 2, GLOBAL_WORK_RESPONSE = 3} REQUEST_TYPE;
	typedef enum { MSG_DATA = 5, MSG_REQUEST = 6} MSG_TAG; //, TAG_TOKEN=12
	typedef enum { ARR = 12, RP = 13} DYN_SCHEME;
/*
	inline std::string get_msg_tag(MSG_TAG tg) {
		switch(tg) {
			case MSG_DATA:
				return "MSG_DATA";
			case MSG_REQUEST:
				return "MSG_REQUEST";
			default:
				std::stringstream ss;
				ss << "UNKNOWN(" << tg << ")" << endl;
				return ss.str();
		}
	} // get_msg_tag

	inline std::string get_request_type(REQUEST_TYPE rt) {
		switch(rt) {
		case RT_WORK_REQUEST:
			return "RT_WORK_REQUEST";
		case RT_WORK_RESPONSE:
			return "RT_WORK_RESPONSE";
		case GLOBAL_WORK_REQUEST:
			return "GLOBAL_WORK_REQUEST";
		case GLOBAL_WORK_RESPONSE:
			return "GLOBAL_WORK_RESPONSE";
		default:
			std::stringstream ss;
			ss << "UNKNOWN(" << rt << ")" << endl;
		return ss.str();
		}
	}
*/

	int num_threads;
	std::vector<int> next_work_request;
	int scheme;
	std::vector<std::deque<int> > message_queue;
	std::vector<std::vector<int> > requested_work_from;
	std::map<int, int> global_work_response_counter;
	int all_idle_work_request_counter;
	omp_lock_t lock;
	std::vector<omp_lock_t*> qlock;

	void init_lb(int num_threads) {
		this->num_threads = num_threads;
		this->scheme = RP;
		next_work_request.clear();
		message_queue.clear();
		requested_work_from.clear();
		for(int i = 0; i < num_threads; i++) {
			int p =  (i + 1) % num_threads;
			next_work_request.push_back(p);
			std::deque<int> dq;
			message_queue.push_back(dq);
			std::vector<int> vc;
			requested_work_from.push_back(vc);
		}
		all_idle_work_request_counter = 0;
	}
/*
	void set_num_threads(int num_threads){
		this->num_threads = num_threads;
	}
	void reinitialize_lb() {
		next_work_request.clear();
		for(int i = 0; i < num_threads; i++) {
			int p =  (i + 1) % num_threads;
			next_work_request.push_back(p);
		}
		all_idle_work_request_counter = 0;
		//TRACE(*logger, "dynamic threads load-balancing REinitialized");
	}
	bool set_load_balancing_scheme(DYN_SCHEME scheme) {
		if( scheme != ARR or scheme != RP)
			return false;
		this->scheme = scheme;
		int thread_id = omp_get_thread_num();
		if(scheme == RP) {
			next_work_request[thread_id] = random() % num_threads;
			while(next_work_request[thread_id] == thread_id)
				next_work_request[thread_id] = random() % num_threads;
		}
	}
	bool all_idle_work_requests_collected() {
		return (all_idle_work_request_counter == num_threads);
	}
	void reset_idle_work_requests_counter() {
		all_idle_work_request_counter = 0;
	}
*/
	//void send_work_request(Thread_private_data &gprv);
	//void send_work_request(Thread_private_data &gprv, int dest);
	//int check_request(Thread_private_data &gprv);
	//void process_request(int source, Thread_private_data &gprv);
	//void process_work_split_request(int source, Thread_private_data &gprv);
	//bool receive_data(int source, int size, Thread_private_data &gprv);
	//void send_msg(int *buffer, int length, int src_thr, int dest_thr);
	//void recv_msg(int *buffer, int length, int thr, int originating_thr);
	void send_work_request(Thread_private_data &gprv) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		if(!requested_work_from[thread_id].empty()) return;
		int buffer[2];
		buffer[0] = RT_WORK_REQUEST;
		buffer[1] = 0;       // filler
		send_msg(buffer, 2, thread_id, next_work_request[thread_id]);
		requested_work_from[thread_id].push_back(next_work_request[thread_id]);
	}
/*
	void send_work_request(Thread_private_data &gprv, int dest) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		if(!requested_work_from[thread_id].empty()) return;
		next_work_request[thread_id] =  dest;
		int buffer[2];
		buffer[0] = RT_WORK_REQUEST;
		buffer[1] = 0; // filler
		send_msg(buffer, 2, thread_id, dest);
		requested_work_from[thread_id].push_back(dest);
	}
*/
	// returns: -1 if no request is made, otherwise rank of the requesting processor
	int check_request(Thread_private_data &gprv) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		if(message_queue[thread_id].size() > 0 ) {
			omp_set_lock(&lock);
			int source = message_queue[thread_id].front();
			omp_unset_lock(&lock);
			return source;
		}
		return -1;
	}
	bool receive_data(int source, int size, Thread_private_data &gprv) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		if(size == 0) {
			if(scheme == RP) {
				next_work_request[thread_id] = random() % num_threads;
				while(next_work_request[thread_id] == thread_id)
					next_work_request[thread_id] = random() % num_threads;
			} else { //ARR
				next_work_request[thread_id] = (next_work_request[thread_id] + 1) % num_threads;
				if(next_work_request[thread_id] == thread_id) //make sure that the request is not made to self
					next_work_request[thread_id] = (next_work_request[thread_id] + 1) % num_threads;
			}
			requested_work_from[thread_id].erase(requested_work_from[thread_id].begin());
			return false;
		}
		if(requested_work_from[thread_id].size() != 1 || requested_work_from[thread_id][0] != source ) {
			exit(1);
		}
		////old comment: nothing else to do, data is already pushed in the queue by the donor thread
		// process the data put in the shared queue
		thread_process_received_data(gprv);

		if(scheme == RP) {
			next_work_request[thread_id] = random() % num_threads;
			while(next_work_request[thread_id] == thread_id)
				next_work_request[thread_id] = random() % num_threads;
		} else { //ARR
			next_work_request[thread_id] = (next_work_request[thread_id] + 1) % num_threads;
			if(next_work_request[thread_id] == thread_id) //make sure that the request is not made to self
				next_work_request[thread_id] = (next_work_request[thread_id] + 1) % num_threads;
		}
		requested_work_from[thread_id].erase(requested_work_from[thread_id].begin());
		thread_start_working();
		return true;
	}
	void process_work_split_request(int source, Thread_private_data &gprv) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		//if threadid = 0 and all threads are idle, no need to send response now
		// if global work is received the response will be sent eventually
		if(thread_id == 0 && all_threads_idle()) {
			all_idle_work_request_counter++;
			return;
		}
		if(thread_working(gprv) == false || can_thread_split_work(gprv) == false) {
			int buffer[2];
			buffer[0] = RT_WORK_RESPONSE;
			buffer[1] = 0;
			send_msg(buffer, 2, thread_id, source);
			return;
		}
		int length;
		thread_split_work(source, length, gprv);
		int buffer_size[2];
		buffer_size[0] = RT_WORK_RESPONSE;
		buffer_size[1] = length; // put there length of the split stack split.size()+1;
		send_msg(buffer_size, 2, thread_id, source);
	}
	void process_request(int source, Thread_private_data &gprv) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		int recv_buf[2];
		recv_msg(recv_buf, 2, thread_id, source);
		switch(recv_buf[0]) {
			case RT_WORK_REQUEST:
				process_work_split_request(source, gprv);
				break;
			case RT_WORK_RESPONSE:
				receive_data(source, recv_buf[1], gprv);
				return;
			//case GLOBAL_WORK_REQUEST:
			//	process_global_work_split_request(recv_buf[1], gprv);
			//	break;
			//case GLOBAL_WORK_RESPONSE:
			//	process_global_work_split_response(recv_buf[1], gprv);
			//	break;
			default:
				exit(1);
		}
	}
	void send_msg(int *buffer, int length, int src_thr, int dest_thr) {
		omp_set_lock(&lock);
		message_queue[dest_thr].push_back(src_thr);
		for(int i = 0; i <length; i++)
			message_queue[dest_thr].push_back(buffer[i]);
		omp_unset_lock(&lock);
	}
	void recv_msg(int *buffer, int length, int thr, int originating_thr) {
		omp_set_lock(&lock);
		int source = message_queue[thr].front();
		if(originating_thr != source) {
			exit(0);
		}
		message_queue[thr].pop_front(); //take off source
		for(int i = 0; i < length; i++) {
			buffer[i] = message_queue[thr].front();
			message_queue[thr].pop_front();
		}
		omp_unset_lock(&lock);
	}
	//hybrid load balancing functions
	//void send_global_split_requests(int requester_rank_id, Thread_private_data &gprv);
	//void process_global_work_split_request(int requester_rank_id, Thread_private_data &gprv);
	//void process_global_work_split_response(int requester_rank_id, Thread_private_data &gprv);
	//void send_work_response_to_all_threads(Thread_private_data &gprv, std::vector<int> thread_has_work);

	// Hybrid Load Balancing Functions starts here
/*
	void send_global_split_requests(int requester_rank_id, Thread_private_data &gprv){
		//send all thread global split request
		int buf[2];
		buf[0] = GLOBAL_WORK_REQUEST;
		buf[1] = requester_rank_id;         // put there length of the split stack split.size()+1;

		for(int i = 0; i< num_threads; i++) {
			if(i != gprv.thread_id)
				send_msg(buf, 2, gprv.thread_id, i);
		}
	}
	void process_global_work_split_request(int requester_rank_id, Thread_private_data &gprv) {
		thread_split_global_work(requester_rank_id, gprv);
		//send global work response to thread 0
		int buf[2];
		buf[0] = GLOBAL_WORK_RESPONSE;
		buf[1] = requester_rank_id;
		send_msg(buf, 2, gprv.thread_id, 0);
	}
	void process_global_work_split_response(int requester_rank_id, Thread_private_data &gprv) {
		if(global_work_response_counter.count(requester_rank_id) == 0)
			global_work_response_counter[requester_rank_id] = 0;
		global_work_response_counter[requester_rank_id]++;
		if(global_work_response_counter[requester_rank_id] == num_threads - 1) {      //exclude itself (thread 0)
			//complete_global_work_split_request(requester_rank_id, gprv); // cxh: disable this for SMP implementation
			global_work_response_counter[requester_rank_id] = 0;
			printf("cxh debug: You shouldn't be here\n");
			exit(1);
		}
	}
	void send_work_response_to_all_threads(Thread_private_data &gprv, std::vector<int> thread_has_work) {
		//send all threads work response
		int buf[2];
		buf[0] = RT_WORK_RESPONSE;

		for(int i = 0; i< num_threads; i++) {
			if(i != gprv.thread_id) {            //0
				buf[1] = thread_has_work[i];       // length, anything but 0 is fine
				send_msg(buf, 2, gprv.thread_id, i);
			}
		}
	}
*/
public:
	lb() { omp_init_lock(&lock); }
	virtual ~lb(){ omp_destroy_lock(&lock); }
	//void threads_load_balance(Thread_private_data &gprv);
	void threads_load_balance(Thread_private_data &gprv) {
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		int src = check_request(gprv);
		if(src != -1)
			process_request(src, gprv);
		if(thread_working(gprv) == false) {
			if(all_threads_idle()) {
				//send_work_request(gprv, 0);
				//printf("cxh debug: All threads idle!\n");
			} else
				send_work_request(gprv);            //global load balance
		}
	}
	//pure virtual functions, to be implemented by the subclass
	virtual bool can_thread_split_work(Thread_private_data &gprv) = 0;
	virtual void thread_split_work(int requesting_thread, int &length, Thread_private_data &gprv) = 0;
	virtual void thread_process_received_data(Thread_private_data &gprv) = 0;
	virtual bool thread_working(Thread_private_data &gprv) = 0;
	virtual bool all_threads_idle() = 0;
	virtual void thread_start_working() = 0;
	//hybrid load balancing functions
	//virtual void thread_split_global_work(int requester_rank_id, Thread_private_data &gprv) = 0;
	//virtual void complete_global_work_split_request(int requester_rank_id, Thread_private_data &gprv) = 0;
};

#endif

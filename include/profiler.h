#pragma once
#include <cstring>
#include <vector>
#include "gardenia.h"
#include "gardenia_io.h"
#include "statistics.h"
#include "env_check.h"
#include "timer.h"

#ifdef ENABLE_PAPI
extern "C" {
#include <papi.h>
#include <papiStdEventDefs.h>
}
#include <iostream>
#endif

#ifdef ENABLE_VTUNE
#include "ittnotify.h"
#endif

#include "timer.h"

namespace profiler {

#ifdef ENABLE_VTUNE
template <typename F>
void profileVtune(const F& func, const char* region) {
  region = region ? region : "(NULL)";
  __itt_resume();
  timeThis(func, region);
  __itt_pause();
}
#else
template <typename F>
void profileVtune(const F& func, const char* region) {
  region = region ? region : "(NULL)";
  std::cout << "Vtune not enabled or found\n";
  timeThis(func, region);
}
#endif

#ifdef ENABLE_PAPI

template <typename __T = void>
void papiInit() {
  /* Initialize the PAPI library */
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    GARDENIA_DIE("PAPI library version mismatch: ", retval, " != ", PAPI_VER_CURRENT);
  }
  if (retval < 0) {
    GARDENIA_DIE("initialization error!");
  }
  //if ((retval = PAPI_thread_init((long unsigned int (*)(void)) pthread_self)) != PAPI_OK) {
  if ((retval = PAPI_thread_init((long unsigned int (*)(void)) omp_get_thread_num)) != PAPI_OK) {
    GARDENIA_DIE("PAPI thread init failed");
  }
}

template <typename V1, typename V2>
void decodePapiEvents(const V1& eventNames, V2& papiEvents) {
  for (size_t i = 0; i < eventNames.size(); ++i) {
    char buf[256];
    std::strcpy(buf, eventNames[i].c_str());
    //std::cout << "event name: " << buf << "\n";
    if (PAPI_event_name_to_code(buf, &papiEvents[i]) != PAPI_OK) {
      GARDENIA_DIE("failed to recognize eventName = ", eventNames[i],
                 ", event code: ", papiEvents[i]);
    }
    //std::cout << "event code: " << papiEvents[i] << "\n";
  }
}

template <typename V1, typename V2, typename V3>
void papiStart(V1& eventSets, V2& papiResults, V3& papiEvents) {
  gardenia::on_each([&](const unsigned tid) {
    //std::cout << "tid: " << tid << "\n";
    int retval;
    if (PAPI_register_thread() != PAPI_OK) {
      GARDENIA_DIE("failed to register thread with PAPI");
    }
    int& eventSet = eventSets[tid];
    eventSet = PAPI_NULL;
    if (PAPI_create_eventset(&eventSet) != PAPI_OK) {
      GARDENIA_DIE("failed to init event set");
    }
    if ((retval = PAPI_add_events(eventSet, papiEvents.data(), papiEvents.size())) != PAPI_OK) {
      std::cerr << "PAPI_add_events error: " << PAPI_strerror(retval) << "\n";
      GARDENIA_DIE("failed to add events");
    }
    /*
    for (size_t i = 0; i < papiEvents.size(); i++ ) {
      //std::cout << "papiEvents[" << i << "]: " << papiEvents[i] << "\n";
      if ((retval = PAPI_add_event(eventSet, papiEvents[i])) != PAPI_OK) {
        std::cerr << "PAPI_add_event error: " << PAPI_strerror(retval) << "\n";
        GARDENIA_DIE("failed to add event");
      }
    }
    //*/
    if (PAPI_start(eventSet) != PAPI_OK) {
      GARDENIA_DIE("failed to start PAPI");
    }
  });
}

template <typename V1, typename V2, typename V3>
void papiStop(V1& eventSets, V2& papiResults, V3& eventNames,
              const char* region) {
  gardenia::on_each([&](const unsigned tid) {
    int& eventSet = eventSets[tid];
    if (PAPI_stop(eventSet, papiResults[tid].data()) != PAPI_OK) {
      GARDENIA_DIE("PAPI_stop failed");
    }
    if (PAPI_cleanup_eventset(eventSet) != PAPI_OK) {
      GARDENIA_DIE("PAPI_cleanup_eventset failed");
    }
    if (PAPI_destroy_eventset(&eventSet) != PAPI_OK) {
      GARDENIA_DIE("PAPI_destroy_eventset failed");
    }
    assert(eventNames.size() == papiResults[tid].size() &&
           "Both vectors should be of equal length");
    for (size_t i = 0; i < eventNames.size(); ++i) {
      reportStat(region, eventNames[i], papiResults[tid][i]);
    }
    if (PAPI_unregister_thread() != PAPI_OK) {
      GARDENIA_DIE("failed to un-register thread with PAPI");
    }
  });
}

template <typename C>
void splitCSVstr(const std::string& inputStr, C& output,
                 const char delim = ',') {
  std::stringstream ss(inputStr);
  for (std::string item; std::getline(ss, item, delim);) {
    output.push_back(item);
  }
}

template <typename F>
void profilePapi(const F& func, const char* region) {
  //int max_nt = omp_get_max_threads();
  //std::cout << "OpenMP maximum " << max_nt << " threads\n";
  std::string ompCSV;
  EnvCheck("OMP_NUM_THREADS", ompCSV);
  int nt = atoi(ompCSV.c_str());
  //assert(nt > 0 && nt <= max_nt);
  //std::cout << "OpenMP using " << nt << " threads\n";

  std::cout << "Profiling with PAPI\n";
  const char* const PAPI_VAR_NAME = "PAPI_EVENTS";
  region                          = region ? region : "(NULL)";
  std::string eventNamesCSV;
  if (!EnvCheck(PAPI_VAR_NAME, eventNamesCSV) || eventNamesCSV.empty()) {
    std::cout << "No Events specified. Set environment variable PAPI_EVENTS, e.g.,\n"
              << "export PAPI_EVENTS=PAPI_L1_DCM\n";
    timeThis(func, region);
    return;
  }
  papiInit();
  std::vector<std::string> eventNames;
  splitCSVstr<std::vector<std::string>>(eventNamesCSV, eventNames);
  auto num_events = eventNames.size();
  std::vector<int> papiEvents(num_events);
  decodePapiEvents<std::vector<std::string>,std::vector<int>>(eventNames, papiEvents);
  std::cout << "Using " << num_events << " event counters: ";
  int i = 0;
  for (auto e : eventNames) {
    std::cout << e << " [" << std::hex << "0x" << papiEvents[i++] << "]  ";
  }
  std::cout << "\n" << std::dec;
  std::vector<int> eventSets(nt);
  std::vector<std::vector<long long>> papiResults(nt);
  for (int tid = 0; tid < nt; tid++) {
    papiResults[tid].resize(num_events);
    std::fill(papiResults[tid].begin(), papiResults[tid].end(), 0);
  }
  papiStart<std::vector<int>,std::vector<std::vector<long long>>,std::vector<int>>
      (eventSets, papiResults, papiEvents);
  timeThis(func, region);
  papiStop<std::vector<int>,std::vector<std::vector<long long>>,std::vector<std::string>>
      (eventSets, papiResults, eventNames, region);
  std::vector<int64_t> counts(num_events);
  for (size_t eid = 0; eid < num_events; eid++) {
    counts[eid] = 0;
    for (int tid = 0; tid < nt; tid++) {
      counts[eid] += papiResults[tid][eid];
    }
    std::cout << eventNames[eid] << " --> " << counts[eid] << "\n";
  }
}
#else
template <typename F>
void profilePapi(const F& func, const char* region) {
  region = region ? region : "(NULL)";
  std::cout << "PAPI not enabled or found\n";
  timeThis(func, region);
}
#endif

} // namespace 


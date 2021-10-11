#include "VertexSet.h"

// VertexSet static members
thread_local std::vector<vidType*> VertexSet::buffers_exist(0);
thread_local std::vector<vidType*> VertexSet::buffers_avail(0);
vidType VertexSet::MAX_DEGREE = 1;

void VertexSet::release_buffers() {
  buffers_avail.clear();
  while(buffers_exist.size() > 0) {
    delete[] buffers_exist.back();
    buffers_exist.pop_back();
  }
}

vidType VertexSet::difference_buf(vidType *outBuf, const VertexSet &other) const {
  vidType ind = bs(ptr,set_size,other.vid);
  //not contained
  VertexSet out;
  for(vidType i=0;i<ind;++i){
    outBuf[i] = ptr[i];
  }
  vidType offset = 0;
  //we have other vid present
  if(ind!=set_size && ptr[ind]==other.vid) {
    offset = 1;
  }
  for(vidType i=ind;i<set_size-offset;++i){
    outBuf[i] = ptr[i+offset];
  }
  return set_size-offset;
}

vidType VertexSet::difference_buf(vidType *outBuf, const VertexSet &other, vidType upper) const {
  vidType indo = bs(ptr,set_size,other.vid);
  vidType indu = bs(ptr,set_size,upper);
  //not contained
  VertexSet out;
  if(indu<=indo) {
    for(vidType i=0;i<indu;++i) {
      outBuf[i] = ptr[i];
    }
    return indu;
  }
  for(vidType i=0;i<indo;++i) {
    outBuf[i] = ptr[i];
  }
  //else, we go up to but not including indu anyway, but that is after indo
  vidType offset = 0;
  //we have other vid present
  if(indo!=set_size && ptr[indo]==other.vid) {
    offset = 1;
  }
  //should actually just be indu TODO verify this
  vidType stop = std::min(indu,set_size);//we go up to, but not including indu
  out.set_size = stop-offset;
  for(vidType i=indo;i<stop-offset;++i) {
    outBuf[i] = ptr[i+offset];
  }
  return stop-offset;
}

vidType VertexSet::difference_ns(const VertexSet &other, vidType upper) const {
  vidType indo = bs(ptr,set_size,other.vid);
  vidType indu = bs(ptr,set_size,upper);
  //other.vid is not in the bounded set
  if(indu<=indo||indo==set_size||ptr[indo]!=other.vid)return indu;
  else return indu-1;
}


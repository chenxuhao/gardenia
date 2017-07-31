/* 
 * use atomicInc to automatically wrap around.
 */

#include <stdio.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "platform_atomics.h"
#define MINCAPACITY	65535
#define MAXOVERFLOWS	1

typedef struct Worklist {
	unsigned pushRange(unsigned *start, unsigned nitems);
	unsigned push(unsigned work);
	unsigned popRange(unsigned *start, unsigned nitems);
	unsigned pop(unsigned &work);
	void clear();
	void myItems(unsigned &start, unsigned &end);
	unsigned getItem(unsigned at);
	unsigned getItemWithin(unsigned at, unsigned hsize);
	unsigned count();

	void init();
	void init(unsigned initialcapacity);
	void setSize(unsigned hsize);
	unsigned getSize();
	void setCapacity(unsigned hcapacity);
	unsigned getCapacity();
	void setInitialSize(unsigned hsize);
	unsigned calculateSize(unsigned hstart, unsigned hend);
	void copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity);
	void append(Worklist wl);

	Worklist();
	~Worklist();
	unsigned ensureSpace(unsigned space);
	unsigned *alloc(unsigned allocsize);
	unsigned realloc(unsigned space);
	unsigned dealloc();
	unsigned freeSize();
	unsigned *items;
	unsigned start, end;
	unsigned capacity;
	unsigned noverflows;
} Worklist;

Worklist::Worklist() {
	init();
}

void Worklist::init() {
	init(0);
}

void Worklist::init(unsigned initialcapacity) {
	setCapacity(initialcapacity);
	setInitialSize(0);
	items = NULL;
	if (initialcapacity) items = alloc(initialcapacity);
	noverflows = 0;
}

unsigned *Worklist::alloc(unsigned allocsize) {
	unsigned *ptr = NULL;
	if(allocsize > 0)
		ptr = (unsigned *)malloc(allocsize * sizeof(unsigned));
	if(ptr == NULL)
		printf("%s(%d): Allocating %d failed.\n", __FILE__, __LINE__, allocsize);
	return ptr;
}

unsigned Worklist::getCapacity() {
	return capacity;
}

unsigned Worklist::calculateSize(unsigned hstart, unsigned hend) {
	if (hend >= hstart) {
		return hend - hstart;
	}
	// circular queue.
	unsigned cap = getCapacity();
	return hend + (cap - hstart + 1);
}

unsigned Worklist::getSize() {
	return calculateSize(start, end);
}

void Worklist::setCapacity(unsigned cap) {
	capacity = cap;
}

void Worklist::setInitialSize(unsigned size) {
	start = 0;
	end = 0;
}

void Worklist::setSize(unsigned size) {
	unsigned cap = getCapacity();
	if (size > cap) {
		printf("%s(%d): buffer overflow, setting size=%d, when capacity=%d.\n", __FILE__, __LINE__, size, cap);
		return;
	}
	if (start + size < cap) {
		end   = start + size;
	} else {
		size -= cap - start;
		end   = size;
	}
}

void Worklist::copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity) {
	if (start < end) {	// no wrap-around.
		memcpy(newitems, olditems + start, oldsize * sizeof(unsigned));
	} else {
		memcpy(newitems, olditems + start, (oldcapacity - start) * sizeof(unsigned));
		memcpy(newitems + (oldcapacity - start), olditems, end * sizeof(unsigned));
	}
}

unsigned Worklist::realloc(unsigned space) {
	unsigned cap = getCapacity();
	unsigned newcapacity = (space > MINCAPACITY ? space : MINCAPACITY);
	if (cap == 0) {
		setCapacity(newcapacity);
		items = alloc(newcapacity);
		if (items == NULL) {
			return 1;
		}
		//printf("\tworklist capacity set to %d.\n", getCapacity());
	} else {
		unsigned *itemsrealloc = alloc(newcapacity);
		if (itemsrealloc == NULL) {
			return 1;
		}
		unsigned oldsize = getSize();
		copyOldToNew(items, itemsrealloc, oldsize, cap);
		dealloc();
		items = itemsrealloc;
		setCapacity(newcapacity);
		start = 0;
		end = oldsize;
		printf("\tworklist capacity reset to %d.\n", getCapacity());
	}
	return 0;
}

unsigned Worklist::freeSize() {
	return getCapacity() - getSize();
}

unsigned Worklist::ensureSpace(unsigned space) {
	if (freeSize() >= space) {
		return 0;
	}
	realloc(space);
	return 1;
}

unsigned Worklist::dealloc() {
	free(items);
	setInitialSize(0);
	return 0;
}

Worklist::~Worklist() {
}

unsigned Worklist::pushRange(unsigned *copyfrom, unsigned nitems) {
	if (copyfrom == NULL || nitems == 0) return 0;

	unsigned lcap = capacity;
	unsigned offset = fetch_and_add(end, nitems);
	assert (offset < lcap); // check overflow.
	for (unsigned ii = 0; ii < nitems; ++ii) {
		items[(offset + ii) % lcap] = copyfrom[ii];
	}
	return 0;
}

unsigned Worklist::push(unsigned work) {
	return pushRange(&work, 1);
}

unsigned Worklist::popRange(unsigned *copyto, unsigned nitems) {
	unsigned currsize = count();
	if (currsize < nitems) {
		nitems = currsize;
	}
	unsigned offset = 0;
	unsigned lcap = capacity;
	if (nitems) {
		if (start + nitems < lcap) {
			offset = fetch_and_add(start, nitems);
		} else {
			offset = fetch_and_add(start, start + nitems - lcap);
		}
	}
	// copy nitems starting from offset.
	for (unsigned ii = 0; ii < nitems; ++ii) {
		copyto[ii] = items[(offset + ii) % lcap];
	}
	return nitems;
}

unsigned Worklist::pop(unsigned &work) {
	return popRange(&work, 1);
}

void Worklist::clear() {
	setSize(0);
}

unsigned Worklist::getItem(unsigned at) {
	unsigned size = count();
	return getItemWithin(at, size);
}

unsigned Worklist::getItemWithin(unsigned at, unsigned size) {
	if (at < size) {
		return items[at];
	}
	return -1;
}

unsigned Worklist::count() {
	if (end >= start) {
		return end - start;
	} else {
		return end + (capacity - start + 1);
	}
}

#define SWAPDEV(a, b)	{ unsigned tmp = a; a = b; b = tmp; }
void printWorklist(Worklist wl) {
	printf("\t");
	for (unsigned ii = wl.start; ii < wl.end; ++ii) {
		printf("%d,", wl.getItem(ii));
	}
	printf("\n");
}

void Worklist::append(Worklist wl) {
	unsigned size = getSize();
	for (unsigned ii = 0; ii < wl.count(); ++ii) {
		items[size + ii] = wl.items[ii];
	}
	end += wl.getSize();
}


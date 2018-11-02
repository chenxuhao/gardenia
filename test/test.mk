# Makefile-based Testing Infrastructure
# Scott Beamer, 2015

# Shared Components ----------------------------------------------------#
#-----------------------------------------------------------------------#

# Dependencies are the tests it will run
test-all: test-build test-load

# Does everthing, intended target for users
test: test-score
	@if $(MAKE) test-score | grep FAIL > /dev/null; \
		then exit 1; \
	fi

# Sums up number of passes and fails
test-score: test-all
	@$(MAKE) test-all | cut -d\  -f 2 | grep 'PASS\|FAIL' | sort | uniq -c

# Result output strings
PASS = \033[92mPASS\033[0m
FAIL = \033[91mFAIL\033[0m

test/out:
	mkdir -p test/out

# Need to be able to build kernels, if this fails rest not run
test-build: all
	@echo " $(PASS) Build"


# Graph Generation/Building/Loading ------------------------------------#
#-----------------------------------------------------------------------#

# Since all implementations use same code for this, only test one kernel
GENERATE_KERNEL = bin/pr_omp_base

# Loading graphs from files
test-load: test-load-pr.mtx

test/out/load-%.out: test/out $(GENERATE_KERNEL)
	./$(GENERATE_KERNEL) test/graphs/$* 1 > $@

.SECONDARY: # want to keep all intermediate files (test outputs)
test-load-%: test/out/load-%.out
	@if grep -q "`cat test/reference/graph-$*.out`" $<; \
		then echo " $(PASS) Load $*"; \
		else echo " $(FAIL) Load $*"; \
	fi

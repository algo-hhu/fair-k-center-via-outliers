# Set the C++ compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Werror -fsanitize=undefined,address -g -MMD -MP -pg

# Define object files and dependencies
OBJECTS = main.o clusteringproblem.o fairlets.o
DEPS = ${OBJECTS:.o=.d}
BIN = fcluster

# Libraries and include paths (optional)
INC_PATHS =
LIB_PATHS =
LIBS = -lemon -lm

# Conditional profiling flag
PROFILING ?= 1
ifeq ($(PROFILING), 1)
    CXXFLAGS += -pg
endif

# Add include paths to CXXFLAGS if defined
CXXFLAGS += ${INC_PATHS}

# Build target
all: ${BIN}

# Include the dependency files
-include ${DEPS}

# Link the object files into the final binary
${BIN}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${LIB_PATHS} -o ${BIN} ${OBJECTS} ${LIBS}

# Clean up build files and profiling output
clean:
	rm -f ${OBJECTS} ${DEPS} ${BIN}

# Declare phony targets
.PHONY: all clean

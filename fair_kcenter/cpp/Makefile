CXX=g++
CXXFLAGS=-std=c++20 -Wall -Werror -fsanitize=undefined,address -g -MMD -MP

OBJECTS=main.o clusteringproblem.o fairlets.o
DEPS=${OBJECTS:.o=.d}
BIN=fcluster

INC_PATHS=
LIB_PATHS=
LIBS=-lemon -lm

CXXFLAGS+= ${INC_PATHS}

all: ${BIN}

-include ${DEPS}

${BIN}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${LIB_PATHS} -o ${BIN} ${OBJECTS} ${LIBS}

clean:
	rm -f ${OBJECTS} ${DEPS} ${BIN}

.PHONY: all clean

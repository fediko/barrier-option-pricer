SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin
INCDIR = ./include

CC = g++

EXEC = main
SRC = $(wildcard $(SRCDIR)/*.cpp main.cpp)
OBJ = $(addprefix $(OBJDIR)/,$(notdir $(SRC:.cpp=.o)))

CXX_FLAGS = -I $(INCDIR) -Wall


.PHONY : compile
compile: $(OBJ)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) -c $(CXX_FLAGS) $< -o $@

$(OBJDIR)/main.o: main.cpp
	$(CC) -c $(CXX_FLAGS) $< -o $@

.PHONY : link
link: compile
	$(CC) -o $(EXEC) $(OBJ) -larmadillo

.PHONY : clean
clean:
	find -type f -name "$(EXEC)" -delete
	find -type f -name "*.o" -delete
	find -type f -name "*~" -delete

.PHONY : all
all: compile link

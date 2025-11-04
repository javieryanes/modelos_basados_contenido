CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
SRC := src/main.cpp
BIN := bin/tfidf

.PHONY: all clean
all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(BIN)

CC = g++
CFLAGS = 

# Nome do executável
TARGET = main.x

# Fontes
SOURCES = image.cpp main.cpp

# Objetos
OBJECTS = $(SOURCES:.cpp=.o)

# Comando pkg-config para OpenCV
OPENCV = `pkg-config --cflags --libs opencv4`

# Caminho para os headers do OpenCV
OPENCV_INCLUDE = -I/usr/include/opencv4

# Regras de compilação
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(OPENCV)

%.o: %.cpp
	$(CC) $(CFLAGS) $(OPENCV_INCLUDE) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJECTS) $(TARGET)

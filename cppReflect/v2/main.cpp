#include "State.h"
#include <iostream>

int main() {
  State state = {1, 2, 0};

  std::cout << "x: " << state.x
    << ", y: " << state. y
    << ", z: " << state. z
    << std::endl;
}

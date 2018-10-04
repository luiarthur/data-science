#include "State.h"
#include <iostream>

struct State {
  int x;
  int y;

  int getField(std::string fieldname) {
    int out;
    if (fieldname == "x") {
      out = x;
    } else {
      out = y;
    }
    return out;
  }
};

int main() {
  State s;
  s.x = 2;
  s.y = 3;

  std::cout << "Field x: " << s.getField("x") << std::endl;
  std::cout << "Field y: " << s.getField("y") << std::endl;
}

#include <string>

struct StateBase {
  template <typename T> T getField(std::string fieldname);
};

#ifndef DOMAIN_H_
#define DOMAIN_H_
#include "real.h"

class Domain {
  private:
    int size;
    Real* data;
    int pos_x, pos_y;

  public:
    Domain(int size_) : size(size_), data(NULL) {}
    ~Domain() {}

    void decompose(int rank, int world_size) {}
};

#endif // DOMAIN_H_

//
// Created by Nicholas Gao on 12/6/17.
//

#include "net.h"

net::net(int numLayers) {
    this->numLayers = numLayers;
}

void net::addLayer() {
    auto l = new layer();
    this->layers.push_back(l);
}
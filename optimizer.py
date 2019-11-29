import numpy as py
import torch


def get_dx_f(x,y):
    return 0


def get_dy_f(x,y):
    return 0


def get_dydy_f(v,x,y):
    return 0


def get_dydx_f(v,x,y):
    return 0


def get_dxdx_f(v,x,y):
    return 0


def get_dxdy_f(v,x,y):
    return 0


def CGD(x, y, dx_f, dy_g, dxdx_f, dxdy_f, dydy_g, dydx_g, eta, gamma):
  dx_f = dx_f(x,y)
  dy_g = dy_g(x,y)
  dx = dx_f - gamma * dxdy_f(dy_g, x, y) + gamma * dxdx_f(dx_f, x, y)
  dy = dy_g - gamma * dydx_g(dx_f, x, y) + gamma * dydy_g(dy_g, x, y)

  x -= eta * dx
  y -= eta * dy
  return x, y

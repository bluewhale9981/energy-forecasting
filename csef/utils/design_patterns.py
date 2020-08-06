# -*- coding: utf-8 -*-


class SingletonDecorator:

    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        if not self.instance:
            self.instance = self.klass(*args, **kwds)
        return self.instance

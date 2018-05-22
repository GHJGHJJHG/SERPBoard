#!/usr/bin/env python3

# logger.py
# Created on 2018-02-07
# Author: Daniel Indictor

import io


class Logger():
  def __init__(self, root = None, indent_style = '\t'):
    if isinstance(root, io.TextIOBase):
      assert root.writable()
    self.root = root
    self.indent_style = indent_style

  def log(self, text):
    if isinstance(self.root, io.TextIOBase):
      self.root.write(text + '\n')
    elif isinstance(self.root, type(self)):
      self.root.log(self.indent_style + text)
    else:
      print(text)

  def __del__(self):
    if isinstance(self.root, io.TextIOBase):
      self.root.close()


if __name__ == '__main__':
  print('Nothing to run!')
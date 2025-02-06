#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:48:43 2025

@author: alexandermikhailov
"""

import re


def trim_string(string: str, fill: str = ' ') -> str:
    return fill.join(filter(lambda _: _, re.split(r'\W', string))).lower()


def validate_input(string: str) -> list[float]:
    return list(map(float, string.split()))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:12:05 2025

@author: alexandermikhailov
"""

from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

DESCRIPTION = {
    'IW': 'Величина сварочного тока',
    'IF': 'Ток фокусировки электронного пучка',
    'VW': 'Скорость сварки',
    'FP': 'Расстояние от поверхности образцов до электронно-оптической системы',
    'Depth': 'Глубина шва',
    'Width': 'Ширина шва'
}

DIMENSIONS = ('Depth', 'Width')

CV = 5

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR.joinpath('data')

MODEL_DIR = BASE_DIR.joinpath('models')

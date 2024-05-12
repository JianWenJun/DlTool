# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @Time:   2024/5/12 21:51
   @Author: ComeOnJian
   @Software: PyCharm
   @File Name:  misc.py
   @Description:
            
-------------------------------------------------
"""
import logging
from typing import Optional

import transformers


def set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

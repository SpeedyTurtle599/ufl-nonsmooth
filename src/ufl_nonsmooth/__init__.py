# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/__init__.py
@brief Shared local non-smooth calculus and native directional propagation.
@details Local laws share values, selected linearisations, and joint directional actions across
    scalar, field, and native symbolic application. Qualified scalar laws provide exact local
    basic coderivatives. Native tape propagation and implicit residuals retain the same input
    activity, checkpoints, boundary conditions, and transpose conventions.
    Import each operation from its defining module.
@author Benjamin Coveler
@date October 2026
"""

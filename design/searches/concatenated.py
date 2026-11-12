"""Reduced CORPSE-in-SK1/BB1: Bando et al. Eqs. (31), (32).

Chronological order, positive constant nominal Rabi rate, variable durations.
Axes use the manuscript convention cos(phi)*sigma_x - sin(phi)*sigma_y.
The phase convention agrees with the existing X-gate benchmark inputs.
"""
import numpy as np


def reduced_corpse(kind, theta, name=None):
    if kind not in ('rCinSK', 'rCinBB'):
        raise ValueError(kind)
    if not 0 < theta < 2*np.pi:
        raise ValueError('Use 0 < theta < 2*pi for the stated CORPSE branch.')
    k = np.arcsin(np.sin(theta/2)/2)
    alpha = np.arccos(-theta/(4*np.pi))
    corpse = [2*np.pi+theta/2-k, 2*np.pi-2*k, theta/2-k]
    if kind == 'rCinSK':
        areas = corpse + [2*np.pi, 2*np.pi]
        phases = [0, np.pi, 0, -alpha, alpha]
    else:
        areas = [np.pi, 2*np.pi, np.pi] + corpse
        phases = [alpha, 3*alpha, alpha, 0, np.pi, 0]
    return dict(name=name or kind, N=len(phases), theta=float(theta), model='drive',
                phases=list(map(float, phases)), amplitudes=[1.]*len(phases),
                durations_T=(np.asarray(areas)/np.pi).tolist(),
                origin='Bando et al., JPSJ 82, 014004 (2013), Eq. '+
                       ('(31)' if kind == 'rCinSK' else '(32)'))


def baseline_controls():
    return {f'{kind}_{target}':reduced_corpse(kind,theta,f'{kind}_{target}')
            for target,theta in [('X',np.pi),('H',np.pi/2)]
            for kind in ['rCinSK','rCinBB']}

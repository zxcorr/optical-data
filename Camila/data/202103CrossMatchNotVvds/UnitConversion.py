#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
versao atualizada em: 25 de marco de 2021
Módulo destinado a conter funcoes que fazem conversao de unidades.
@author: camila

"""

def hmstodegree(array):
    """ Função que converte uma array com coordenadas em unidades de tempo que normalmente
    caracterizam uma Ascenção Reta(RA)para grau decimal
    input = array com dados do tipo strings onde horas, minutos e segundos estão separados por
    um espaço
    output = array com dados do tipo float em graus decimais"""
    array_graus = []
    for i in range(len(array)):
        hms = array[i].split()
        # se o array hms conter 2 elementos, adicionar um terceiro elemento 0, para evitar erros do tipo 'array out of range'
        if len(hms) == 2: 
            hms.append(0)
        h = float(hms[0])
        m = float(hms[1])
        s = float(hms[2])
        # equacao de conversao
        graus = h*15 + m*(15/60) + s*(15/3600)
        array_graus.append(graus)
        
    return array_graus

def dmstodegree(array):
    """ Função que converte uma array com coordenadas em unidades angulares que normalmente
    caracterizam uma Declinação(DEC) para grau decimal
    input = array com dados do tipo strings onde graus, minutos e segundos estão separados por
    um espaço
    output = array com dados do tipo float em graus decimais"""
    array_graus = []
    for i in range(len(array)):
        dms = array[i].split()
        # se o array dms conter 2 elementos, adicionar um terceiro elemento 0, para evitar erros do tipo 'array out of range'
        if len(dms) == 2:
            dms.append(0)
        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])
        if d < 0:
            graus = d - m*(1/60) - s*(1/3600)
        else:
        	#equacao de conversao
            graus = d + m*(1/60) + s*(1/3600)
        array_graus.append(graus)
    return array_graus

#! /usr/bin/env python
#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
import locale
locale.setlocale(locale.LC_NUMERIC, "")
def format_num(num):
    """Format a number according to given places.
    Adds commas, etc. Will truncate floats into ints!"""

    try:
        fnum = float(num)
        return locale.format("%.*f", fnum, True)

    except (ValueError, TypeError):
        return str(num)

def get_max_width(table, index):
    """Get the maximum width of the given column index"""
    return max([len(format_num(row[index])) for row in table])


def make_table_from_dictionary(table, columns=[]):
    """Creates a list-based table from the given table[col] = [row]"""
    new_table=[]

    if len(columns) > 0:
        cols=[]
        for c in columns:
            cols += [c]
        new_table += [cols]
    
    for c in table:
        row=[c]
        for i in table[c]:
            row += [i]
        
        new_table += [row]

    return new_table

def print_table(table):
    """Prints out a table of data, padded for alignment
    @param table: The table to print. A list of lists.
    Each row must have the same number of columns. """
    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))
    
    for row in table:
        # left col
        print row[0].ljust(col_paddings[0] + 1),
        # rest of the cols
        for i in range(1, len(row)):
            col = format_num(row[i]).rjust(col_paddings[i] + 2)
            print col,
        print


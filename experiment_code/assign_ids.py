#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:20:42 2025

@author: max
"""

import csv
import tkinter as tk
from tkinter import simpledialog, messagebox

def assign_id():
    try:
        # Read CSV
        with open('participant_ID_list.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        # Get participant number
        root = tk.Tk()
        root.withdraw()
        seq_num = simpledialog.askinteger("Input", "Enter participant's sequential number:", parent=root)
        
        if seq_num is None:
            return
            
        # Validate input
        if seq_num < 1 or seq_num > len(rows):
            messagebox.showerror("Error", "Invalid sequential number")
            return
            
        # Check if already used
        if rows[seq_num-1]['used'] == '1':
            messagebox.showerror("Error", f"ID for participant {seq_num} already assigned")
            return
            
        # Get ID and update
        subject_id = rows[seq_num-1]['subject_id']
        rows[seq_num-1]['used'] = '1'
        
        # Write back to CSV
        with open('participant_ID_list.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject_id', 'used'])
            writer.writeheader()
            writer.writerows(rows)
            
        # Show result
        messagebox.showinfo("Assigned ID", 
            f"Participant {seq_num}\nAssigned ID: {subject_id}")
            
    except FileNotFoundError:
        messagebox.showerror("Error", "participant_ID_list.csv not found")
    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    assign_id()
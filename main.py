import tkinter as tk
from tkinter import ttk, messagebox
from genetic_algo import SeatingOptimizer

class SeatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classroom Seating Optimizer")
        
        # Initialize variables
        self.num_students = tk.StringVar()
        self.num_rows = tk.StringVar()  # Added this
        self.num_cols = tk.StringVar()  # Added this
        
        # Set default values
        self.num_students.set("20")
        self.num_rows.set("5")    # Added this
        self.num_cols.set("6")    # Added this
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add input fields
        # Students
        ttk.Label(main_frame, text="Number of Students:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(main_frame, textvariable=self.num_students).grid(row=0, column=1, padx=5, pady=5)
        
        # Rows
        ttk.Label(main_frame, text="Number of Rows:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(main_frame, textvariable=self.num_rows).grid(row=1, column=1, padx=5, pady=5)
        
        # Columns
        ttk.Label(main_frame, text="Number of Columns:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(main_frame, textvariable=self.num_cols).grid(row=2, column=1, padx=5, pady=5)
        
        # Add optimize button
        ttk.Button(main_frame, text="Optimize", 
                  command=self.optimize).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Add result label
        self.result_label = ttk.Label(main_frame, text="")
        self.result_label.grid(row=4, column=0, columnspan=2)


    # Add to the SeatingApp class in main.py
    def create_progress_window(self):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Optimization Progress")
        
        # Progress label
        self.progress_label = ttk.Label(progress_window, text="Generation: 0")
        self.progress_label.pack(pady=10)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_window, 
            length=300, 
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        return progress_window

    def update_progress(self, generation, max_generations):
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text=f"Generation: {generation}")
            progress = (generation / max_generations) * 100
            self.progress_bar['value'] = progress
            self.root.update()

    def update_progress(self, generation, max_generations):
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text=f"Generation: {generation}")
            progress = (generation / max_generations) * 100
            self.progress_bar['value'] = progress
            self.root.update()
    
    def optimize(self):
        try:
            # Convert and validate inputs
            students = int(float(self.num_students.get()))
            rows = int(float(self.num_rows.get()))
            cols = int(float(self.num_cols.get()))
            
            # Validate ranges
            if students <= 0 or rows <= 0 or cols <= 0:
                messagebox.showerror("Error", "Please enter positive numbers!")
                return
                
            if students > rows * cols:
                messagebox.showerror("Error", 
                                f"Too many students ({students}) for room size ({rows}x{cols})!")
                return
            
            # Create optimizer with feedback
            self.result_label.config(text="Optimization in progress...")
            self.root.update()  # Update GUI to show progress message
            
            optimizer = SeatingOptimizer(students, rows, cols)
            
            # Run optimization with progress updates
            best_layout, best_fitness = optimizer.optimize()
            
            # Show results
            self.result_label.config(
                text=f"Optimization completed!\nBest fitness: {best_fitness:.2f}"
            )
            
            # Display the seating arrangement
            self.display_seating(best_layout)
            
        except ValueError as ve:
            messagebox.showerror("Error", "Please enter valid numbers!")
            print(f"Value Error: {ve}")  # For debugging
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error: {e}")  # For debugging


    def display_seating(self, layout):
        seating_window = tk.Toplevel(self.root)
        seating_window.title("Seating Arrangement")
        
        # Create a frame for the seating grid
        grid_frame = ttk.Frame(seating_window, padding="10")
        grid_frame.grid(row=0, column=0, sticky="nsew")
        
        # Color scheme for visualization
        colors = {
            0: "#ffffff",  # Empty seat - white
            1: "#90EE90"   # Occupied seat - light green
        }
        
        # Create grid of seats
        for i in range(layout.shape[0]):
            for j in range(layout.shape[1]):
                seat_frame = ttk.Frame(
                    grid_frame, 
                    borderwidth=2, 
                    relief='solid'
                )
                seat_frame.grid(
                    row=i, column=j, 
                    padx=2, pady=2, 
                    sticky='nsew'
                )
                
                # Get seat value
                value = int(layout[i, j])
                
                # Create label with background color
                label = tk.Label(
                    seat_frame,
                    text=f"S{value}" if value > 0 else "",
                    width=4, height=2,
                    bg=colors[1 if value > 0 else 0],
                    relief='raised' if value > 0 else 'sunken'
                )
                label.pack(expand=True, fill='both')
                
                # Add tooltip with student info if seat is occupied
                if value > 0:
                    self.create_tooltip(label, f"Student {value}\nRow {i+1}, Col {j+1}")
        
        # Add legend
        legend_frame = ttk.Frame(seating_window, padding="10")
        legend_frame.grid(row=1, column=0, sticky="ew")
        
        ttk.Label(legend_frame, text="Empty Seat:", padding="5").grid(row=0, column=0)
        tk.Label(legend_frame, bg=colors[0], width=2).grid(row=0, column=1)
        
        ttk.Label(legend_frame, text="Occupied Seat:", padding="5").grid(row=0, column=2)
        tk.Label(legend_frame, bg=colors[1], width=2).grid(row=0, column=3)

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0")
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
        
        widget.bind('<Enter>', show_tooltip)
        def create_tooltip(self, widget, text):
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                
                label = ttk.Label(tooltip, text=text, background="#ffffe0")
                label.pack()
                
                def hide_tooltip():
                    tooltip.destroy()
                
                widget.tooltip = tooltip
                widget.bind('<Leave>', lambda e: hide_tooltip())
            
            widget.bind('<Enter>', show_tooltip)

if __name__ == "__main__":
    root = tk.Tk()
    app = SeatingApp(root)
    root.mainloop()
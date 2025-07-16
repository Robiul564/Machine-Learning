import cv2
import random
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")
print("Model class names:", model.names)

# Choices
choices = ["rock", "paper", "scissors"]

# Game state
user_score = 0
computer_score = 0
round_count = 0
max_rounds = 10
game_over = False

# Decide winner of each round
def decide_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Draw!"
    elif (user_choice == "rock" and computer_choice == "scissors") or \
         (user_choice == "paper" and computer_choice == "rock") or \
         (user_choice == "scissors" and computer_choice == "paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

# Capture webcam
cap = cv2.VideoCapture(0)

# GUI
root = tk.Tk()
root.title("Rock Paper Scissors Game")
root.geometry("800x600")

# Video display label
lmain = tk.Label(root)
lmain.pack()

# Info label
info_label = tk.Label(root, text="Get ready and press 'Play Round'!", font=("Arial", 16))
info_label.pack()

# Score label
score_label = tk.Label(root, text="Your Score: 0 | Computer Score: 0 | Round: 0/10", font=("Arial", 14))
score_label.pack()

# Function to update video
def show_frame():
    if not game_over:
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
    root.after(10, show_frame)

# Play round function
def play_round():
    global user_score, computer_score, round_count, game_over

    ret, frame = cap.read()
    if not ret or game_over:
        return

    # Predict using model
    results = model(frame)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        class_id = int(boxes.cls[0].item())
        user_choice = model.names[class_id].lower()
        computer_choice = random.choice(choices)

        result = decide_winner(user_choice, computer_choice)

        # Update scores
        if result == "You Win!":
            user_score += 1
        elif result == "Computer Wins!":
            computer_score += 1

        round_count += 1

        # Update text
        info_label.config(text=f"You: {user_choice}, Computer: {computer_choice}\nResult: {result}")
        score_label.config(text=f"Your Score: {user_score} | Computer Score: {computer_score} | Round: {round_count}/{max_rounds}")

        # Check if game over
        if round_count >= max_rounds:
            game_over = True
            if user_score > computer_score:
                final_text = "🎉 You are the overall winner!"
            elif user_score < computer_score:
                final_text = "😢 Computer wins the game!"
            else:
                final_text = "🤝 It's a draw overall!"

            info_label.config(text=final_text)
    else:
        info_label.config(text="No hand detected! Try again.")

# Play button
play_button = tk.Button(root, text="Play Round", font=("Arial", 16), command=play_round)
play_button.pack()

# Close everything when window closes
def on_closing():
    global game_over
    game_over = True
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start video loop
show_frame()
root.mainloop()

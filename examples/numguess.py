import random

print("🎲 " + "="*58 + " 🎲")
print("   Number Guessing Game - The Simplest RL Example")
print("🎲 " + "="*58 + " 🎲")

# Environment setup
target = random.randint(1, 10)
guesses_left = 3

print(f"\n🎯 I'm thinking of a number between 1 and 10...")
print(f"💭 You have {guesses_left} guesses. Let's see how random guessing works!\n")

# The RL Loop - Pure random policy (no learning!)
while guesses_left > 0:
    # Policy: Random guessing (no learning yet!)
    guess = random.randint(1, 10)
    guesses_left -= 1
    
    print(f"💭 Guess #{3-guesses_left}: {guess}", end=" → ")
    
    # Reward signal (but we're not using it!)
    if guess == target:
        print("🎉 Correct! +10 points")
        break
    elif abs(guess - target) <= 2:
        print("🔥 Warm! (close)")
    else:
        print("❄️  Cold! (far)")
else:
    print(f"\n💔 Out of guesses. The number was {target}.")

print("\n" + "="*62)
print("💡 This is RL: Observe → Act → Reward → Repeat")
print("   But this policy is terrible! It doesn't learn from rewards.")
print("="*62 + "\n")
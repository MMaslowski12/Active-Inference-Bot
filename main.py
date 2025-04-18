"""
Active Inference Bot - Main Entry Point

This project demonstrates active inference in different scenarios:
1. Demo - A simple 1D world with linear/quadratic functions
2. Maze - A 2D maze environment with discrete states

To run a specific demo:
1. Demo: python -m applications.demo.main
2. Maze: python -m applications.maze.main
"""

def main():
    print("Please run one of the following commands:")
    print("1. Demo: python -m applications.demo.main")
    print("2. Maze: python -m applications.maze.main")

if __name__ == "__main__":
    main()


"""
This session:
-- Implement the MazeAgent with its own probability models
-- See if VFE minimalization easily translates to the maze

Next session:
-- Fix up anything from last session if there's a problem
-- Work on EFE <-- map it out

The next next session:
-- Implement EFE for demo
-- Implement EFE for the Maze
-- Publish

Next x3 session
-- Implement it for OvAI


For now: just the position.
y = position, 
p(y|x) = Identity matrix
p(x) = 0.2 * vector

-- Create Discrete distribution
-- Set up the p(y|x) and p(x) this way
-- Crate the maze loop
-- See the Q


"""
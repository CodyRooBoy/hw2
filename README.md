# Homework 2: Explore Exploit

### Group Members
    - Brandon Herrin (A02336477)
    - Josh Weeks (A02304519)

### Problems:

1. **Epsilon-Greedy Method:**
   - Identify the **optimal epsilon** that provides the fastest convergence.:
     - it is incredibly hard to tell. For most of our runs, the optimal epsilon was usually either .01 or .05, but if we had to choose one it would be .05

2. **Thompson Sampling Algorithm:**
   - Compare the **convergence speed and performance** of Thompson Sampling with the best epsilon-greedy result:
   - **Convergence**:
     - Thompson Sampling converges slower than the fastest epsilon greedy method.
    - **Performance**:
        - Thompson Sampling outperforms/performs just as well as the best epsilon greedy method. 
        - if performance is meant to measure the **SPEED** of the algorithm on our hardware, then thompson performed much slower. This is likely due to the fact that thompson sampling requires building beta distributions for each machine.

3. **Quenching Functions in Epsilon-Greedy:**
   - Compare the **convergence rates** for each quenching strategy.
   - Analyze which quenching function provides the best balance of exploration and exploitation, and **explain why**.
    - **Convergence**:
        - for our runs, heavy asymptotic quenching converged fastest out of all three methods.
    - **Analyze**:
       - heavy asymptotic quenching performed best out of all three methods. Linear quenching performed worst out of all three methods, it took too long to find converge. Asymptotic quenching performed fairly well, much better than linear. 
       - Heavy asymptotic performed best, this is likely because it is encouraged to explore heavily early, to gather information, and then tapers off heavily. 
        

4. **Exploration Strategy Modifications:**
   - Implement **alternative exploration strategies** and compare their results with the original epsilon-greedy approach.
   - **exclude best**:
       - exclude best perfomed better than the original epsilon greedy. It had the same story as before, it was hard to tell. For most of our runs, the optimal epsilon was usually either .01 or .05, but if we had to choose one it would be .05
    - **weighted exploration**:
        - weighted perfomed better than the original epsilon greedy. Once again, depending on the run the optimal epsilon was either .05 or .01. sometimes even .1 for this strategy.
   - Analyze the **convergence speed and accuracy** of these strategies:
        - **convergence**: 
            - Weighted exploration seemed to converge fastest
        - **accuracy**:
           - They both perfomed similarly. typically the optimal epsilon for each strategy earned a similair running average. close to 2, which is the average of both the best machines.

   - Reflect on how these modifications affect the balance between **exploration and exploitation**.
   - **Exploration**:
        - Weighted exploration seems to be more apt at exploration, however exlclude best is better than the original epsilon greedy.
    - **Exploitation**:
        - Weighted exploration likely had an edge, but it was close. Weighted exploration is an engenious strategy. If there isn't enough information, more is explored until satisfied. 

**Part 3**
    - **Reflection**:
        - Randomness helps convergance under dynamic conditions by giving more opportunities for epsilon-greedy and Thompson Sampling to change their selection and find other "bandits" that may be more ideal. The drawback of this is that the algorithms could fail to commit to an action and continue exploring.
        - The dynamic adjustments is more adaptable than the standard Epsilon-Greedy algorithm. Shifts in the modified algorithm encourage exploration unlike the standard approach which explores with a fixed probability.
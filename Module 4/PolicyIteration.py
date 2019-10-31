import gridworld_mdp as gw

def policy_iteration(state_count, gamma, theta, get_available_actions, get_transitions):
    """
    This function computes the optimal value function and policy for the specified MDP, using the Policy Iteration algorithm.
    'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
    
    'gamma' is the MDP discount factor for rewards.
    
    'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
    
    'get_available_actions' returns a list of the MDP available actions for the specified state parameter.
    
    'get_transitions' is the MDP state / reward transiton function.  It accepts two parameters, state and action, and returns
        a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).  
    """
    V = state_count*[0]                # init all state value estimates to 0
    pi = state_count*[0]
    nA = len(get_available_actions(0))
    prob_pi_act = [[1/nA, 1/nA, 1/nA, 1/nA]]*state_count
    # init with a policy with first avail action for each state
    for s in range(state_count):
        avail_actions = get_available_actions(s)
        pi[s] = avail_actions[0]
    while True:
        while True:
            delta = 0
            for i in range(1,state_count):
                tt = 0
                avail_actions = get_available_actions(i)
                for act in avail_actions:
                    next_state, reward, prob = get_transitions(state=i,action=act)[0]
                    tt += prob_pi_act[i][avail_actions.index(act)]*prob*(reward + gamma*V[next_state])
                delta = max(delta,abs(tt - V[i]))
                V[i] = tt
            if delta < theta:
                break
                
        policy_stable = True
        for i in range(1,state_count):
            old_action = pi[i]
            tmp = -9999
            avail_actions = get_available_actions(i)
            for act in avail_actions:
                next_state, reward, prob = get_transitions(state=i,action=act)[0]
                _tmp = prob*(reward + gamma*V[next_state])
                if tmp < _tmp:
                    tmp = _tmp
                    pi[i] = act
            if old_action != pi[i]:
                policy_stable = False
            __tmp = [0]*nA
            __tmp[avail_actions.index(pi[i])] = 1
            prob_pi_act[i] = __tmp
        if policy_stable:
            break

    # insert code here to iterate using policy evaluation and policy improvement (see Policy Iteration algorithm)
    return (V, pi)        # return both the final value function and the final policy


n_states = gw.get_state_count()

# test our function
values, policy = policy_iteration(state_count=n_states, gamma=.9, theta=.001, get_available_actions=gw.get_available_actions, \
    get_transitions=gw.get_transitions)

print("Values=", values)
print("Policy=", policy)
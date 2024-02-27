import numpy as np
import math

#Function for getting all the happiness levels
def happiness_level(pref: np.array, outcome: str) -> np.array:
    m, n = pref.shape
    new_voting = np.array([[pref[i][j] for i in range(m)] for j in range(n)])
    h = list()
    for ivoter in range(n):
        ind_pref = new_voting[ivoter]
        h.append(ind_happiness(ind_pref, outcome, m))
    return h

#Function for getting the happiness of just one individual
def ind_happiness(ind_pref: np.array, outcome: str, m: int) -> float:
    d = np.where(ind_pref==outcome)[0][0]
    h = distr_h(d, m)
    return h
    
#Distribution of h based on d (position of the option that won)
def distr_h(d, m):
    h_i = (1-2/(m-1)*d)
    k = 0.95
    c = 1/(2*math.atanh(k))
    h = math.atanh(h_i*k)*c+0.5
    return h

def distr_h_straight(d, m):
    h_i = (1-1/(m-1)*d)
    return h_i

def main():

    #The main function is supposed to plot the possible values of happiness based on the amount of possible voting options.
    #Also, it prints the different happiness levels for each voter
    import warnings

    warnings.filterwarnings("ignore")

    import matplotlib.pyplot as plt

    voting = np.array([['C', 'B', 'C', 'B', 'B'],
       ['A', 'D', 'D', 'D', 'A'],
       ['D', 'C', 'A', 'C', 'D'],
       ['B', 'A', 'B', 'A', 'C']])
    h = happiness_level(voting, 'C')
    print(h)
    m = 6

    x_values = np.linspace(0, 99, 100)
    y_values = list()
    h_x_values = np.linspace(0, m-1, m)
    h_y_values = list()

    for i in range(len(x_values)):
        y_values.append(distr_h(x_values[i], 100))
        x_values[i] = x_values[i] / 99 * (m-1) 
    for x in h_x_values:
        h_y_values.append(distr_h(x, m))

    plt.plot(x_values, y_values)
    plt.plot(h_x_values, h_y_values, 'o')
    plt.xlabel('d')
    plt.ylabel('h')
    plt.title('Happiness Level')

    # Show the plot
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()




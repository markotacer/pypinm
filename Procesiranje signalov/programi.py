def limiti(kanal):
    '''Vzame podatek o kanalu, ki ga obravnava
    Vrne Povprečno vrednost spodnjih in zgornjih limit'''
    peaks1, _ = find_peaks(kanal, distance=4)
    lows1, _ = find_peaks(-kanal, distance=4)
    A2 = np.average(kanal[peaks1])
    A1 = np.average(kanal[lows1])
    return A1, A2
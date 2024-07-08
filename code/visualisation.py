import pyqtgraph as pg
import numpy as np

def nk_plot(wavelength, n, k = None):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()
    #eV = np.array(eV).reshape(151)
    n = np.array(n)
    plt.setLabel('bottom', 'Wavelength', units='μm')
    plt.plot(wavelength, n, pen=pg.mkPen('red', width=1.5), name='n')
    if k is not None:
        plt.plot(wavelength, k, pen=pg.mkPen('blue', width=1.5), name='k')

    plt.setWindowTitle('Refractive index plot')
    plt.setBackground('w')
    

def e_plot(ph_energy, e1, e2):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()
    plt.setLabel('bottom', 'Photon energy', units='eV')
    plt.plot(ph_energy, e1, pen=pg.mkPen('red', width=1.5), name='e1')
    plt.plot(ph_energy, e2, pen=pg.mkPen('green', width=1.5), name='e2')
    '''if wavelength:
    else:
        plt.setLabel('bottom', 'Photon energy', units='eV')
        plt.plot(eV, n, pen=pg.mkPen('red', width=1.5), name='n')
        if k is not None:
            plt.plot(eV, k, pen=pg.mkPen('blue', width=1.5), name='k')

    '''
    plt.setWindowTitle('e plot')
    plt.setBackground('w')

def compare_results(x_space, y_pred, y_true):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()
    #plt.setLabel('bottom', 'Wavelength')
    if len(y_pred.shape) > 1:
        for i in range(y_pred.shape[1]):
            plt.plot(x_space, y_pred[:, i], pen=pg.mkPen(color=(i), width=1.5), name=f'prediction {i}')
    else:
        plt.plot(x_space, y_pred, pen=pg.mkPen('red', width=1.5), name=f'prediction')

    if len(list(y_true.shape)) > 1:
        for j in range(y_true.shape[1]):
            plt.plot(x_space, y_true.iloc[:, j], pen=pg.mkPen((i+1+j), width=1.5), name=f'true {j}')
    else:
        plt.plot(x_space, y_true, pen=pg.mkPen('blue', width=1.5), name=f'true')
    

    plt.setWindowTitle('Refractive index plot')
    plt.setBackground('w')
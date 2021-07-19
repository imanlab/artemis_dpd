import roslibpy
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


def receive_sensor(msg):
    global sensor
    sensor = msg

if __name__ == "__main__":

    sensor = None

    client = roslibpy.Ros(host='127.0.0.1', port=9090)
    client.run()
    sub_data = roslibpy.Topic(client, '/xServTopic', 'xela_server/XStream')

# PLOT AXIS
    n = 24
    m = 36
    y = np.linspace(0, 3, 4)
    x = np.linspace(0, 5, 6)
    y2 = np.linspace(0, 3, n)
    x2 = np.linspace(0, 5, m)

    XX, YY = np.meshgrid(x, y)
    XX = np.reshape(XX, -1)
    # YY = np.flip(np.reshape(YY, -1), 0)
    YY = np.reshape(YY, -1)
    XX2, YY2 = np.meshgrid(x2, y2)

    comp_vect = np.ones(3 * 24)
    comp_vect[0:24] = 70
    comp_vect[24:48] = 30
    comp_vect[48::] = 30

    j = 0

    Z0 = np.ones(24 * 3)
    ZZ = np.ones(24 * 3)

    Z2 = np.zeros(n*m)
    X2 = np.zeros(n*m)
    Y2 = np.zeros(n*m)

    Z = ZZ[0:24]
    X = ZZ[24:48]
    Y = ZZ[48::]

    f_Z = interp2d(XX, YY, Z, kind='linear')
    Z2 = f_Z(x2, y2)

    f_X = interp2d(XX, YY, X, kind='linear')
    X2 = f_X(x2, y2)

    f_Y = interp2d(XX, YY, Y, kind='linear')
    Y2 = f_Y(x2, y2)


    fig, ax_lst = plt.subplots(figsize=(13, 3), ncols=3)
    ax_lst = ax_lst.ravel()
    plt.show(block=False)


    im1 = ax_lst[0].imshow(Z2, interpolation='sinc',
                          origin='bottom',
                          vmin=0,
                          vmax=3000,
                          cmap='Spectral')

    im2 = ax_lst[1].imshow(X2, interpolation='bicubic',
                          origin='bottom',
                          aspect='equal',
                          vmin=0,
                          vmax=700,
                          cmap='Spectral')

    im3 = ax_lst[2].imshow(Z2, interpolation='bicubic',
                          origin='bottom',
                          aspect='equal',
                          vmin=0,
                          vmax=700,
                          cmap='Spectral')

    ax_lst[0].title.set_text('Normal Force')
    ax_lst[1].title.set_text('Shear Force along x')
    ax_lst[2].title.set_text('Shear Force along y')
    fig.colorbar(im1, ax=ax_lst[0], shrink=0.5)
    fig.colorbar(im2, ax=ax_lst[1], shrink=0.5)
    fig.colorbar(im3, ax=ax_lst[2],shrink=0.5)

    while True:

        sub_data.subscribe(receive_sensor)

        if sensor!=None:


            if j == 0:
                for i in range(24):

                    Z0[i] = np.array(sensor['data'][0]['xyz'][i]['z'])
                    Z0[i + 24] = np.array(sensor['data'][0]['xyz'][i]['x'])
                    Z0[i + 48] = np.array(sensor['data'][0]['xyz'][i]['y'])
                j=+1

            for i in range(24):
                ZZ[i] = np.array(sensor['data'][0]['xyz'][i]['z'])
                ZZ[i + 24] = np.array(sensor['data'][0]['xyz'][i]['x'])
                ZZ[i + 48] = np.array(sensor['data'][0]['xyz'][i]['y'])

            ZZ = (ZZ - Z0) * (abs(ZZ - Z0) > comp_vect)
            Z = ZZ[0:24]
            X = ZZ[24:48]
            Y = ZZ[48::]


            f_Z = interp2d(XX, YY, Z, kind='linear')
            Z2 = f_Z(x2, y2)

            f_X = interp2d(XX, YY, X, kind='linear')
            X2 = f_X(x2, y2)

            f_Y = interp2d(XX, YY, Y, kind='linear')
            Y2 = f_Y(x2, y2)

            im1.set_data(Z2)
            im2.set_data(X2)
            im3.set_data(Y2)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

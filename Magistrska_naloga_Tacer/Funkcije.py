def Admitanca(w, omega, Phi, modal_shapes_l_limit = 0, modal_shapes_u_limit = 2,o_l_limit = 0, o_u_limit = 4, i_l_limit = 0, i_u_limit = 3):
    """Funkcjia vzame vektor lastnih vrednosti w in Normiran vektor Lastnih oblik. Z definicijo mej določimo kaj je zajeto kot input in output

    Args:
        w (vector): vektor lastnih vrednosti
        omega (vector): vektor obravnavanih frekvenc
        Phi (matrix): kvadratna matrika lastnih oblik
        modal_shapes_l_limit (int, optional): spodnja meja upoštevanih modeshapes. Defaults to 0.
        modal_shapes_u_limit (int, optional): zgornja meja upoštevanih modeshapes. Defaults to 2.
        o_l_limit (int, optional): Output lower limit. Defaults to 0.
        o_u_limit (int, optional): Output upperr limit. Defaults to 4.
        i_l_limit (int, optional): Input lower limit. Defaults to 0.
        i_u_limit (int, optional): Input upperr limit. Defaults to 3.
    """
    
    omega_r = w[modal_shapes_l_limit:modal_shapes_u_limit] # lastne frekvence
    Phi_output = Phi[o_l_limit:o_u_limit,modal_shapes_l_limit:modal_shapes_u_limit].copy()
    Phi_input = Phi[i_l_limit:i_u_limit, modal_shapes_l_limit:modal_shapes_u_limit].copy()
    
    Y_2_einsum = np.einsum('om, im, fm -> foi', Phi_output, Phi_input, 1. / (-omega[:,None]**2 + omega_r**2))
    print(f'Admitanca -> shape: {Y_2_einsum.shape}')
    return Y_2_einsum
    
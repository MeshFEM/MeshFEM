import energy

nu = 0.3
K = 1

def get2DEnergy(energy_type, l, mu, smooth_inversion_penalty = False):
    if energy_type == energy.EnergyType.LINEAR:
        tensor = energy.ElasticityTensor2D()
        tensor.setIsotropic(K, nu)
        return energy.LinearElasticEnergy2D(tensor)
    elif energy_type == energy.EnergyType.NEO_HOOKEAN:
        print (l)
        print (mu)
        return energy.NeoHookeanEnergy2D(l, mu, smooth_inversion_penalty)
    return None

def get3DEnergy(energy_type, l, mu, smooth_inversion_penalty = False):
    if energy_type == energy.EnergyType.LINEAR:
        tensor = energy.ElasticityTensor3D()
        tensor.setIsotropic(K, nu)
        return energy.LinearElasticEnergy3D(tensor)
    elif energy_type == energy.EnergyType.NEO_HOOKEAN:
        return energy.NeoHookeanEnergy3D(l, mu, smooth_inversion_penalty)
    return None


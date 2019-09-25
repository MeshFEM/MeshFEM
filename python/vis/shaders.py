import pythreejs, os
from types import MethodType

directory = os.path.dirname(os.path.realpath(__file__)) + '/shaders'

def loadShaderMaterial(name):
    uniforms, lights = None, True
    vs, fs = None, None
    if (name == 'vector_field'):
        uniforms = dict(
                    arrowAlignment=dict(value=-1.0),
                    **pythreejs.UniformsLib['lights'],
                    **pythreejs.UniformsLib['common']
                )
        vs = open('{}/{}.vert'.format(directory, name), 'r').read()
        fs = open('{}/{}.frag'.format(directory, name), 'r').read()
    else:
        raise Exception('Unknown shader : ' + name)

    mat =  pythreejs.ShaderMaterial(
                uniforms=uniforms,
                vertexShader=vs,
                fragmentShader=fs,
                side='DoubleSide',
                lights=lights)

    # Attach an updateUniforms for more easily changing the uniforms of a shader material
    def updateUniforms(sm, **kwargs):
        u = dict(**sm.uniforms)
        u.update({k: dict(value=v) for k, v in kwargs.items()})
        sm.uniforms = u
        sm.needsUpdate = True
    mat.updateUniforms = MethodType(updateUniforms, mat)

    return mat

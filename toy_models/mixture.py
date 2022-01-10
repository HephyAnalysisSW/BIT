import ROOT
import numpy as np
from math import exp
from operator import mul
import itertools

class ComponentPDF:
    def __init__(self):
        pass
    @staticmethod
    def product_parameters( alpha, beta):
        return tuple(map(sum, zip(alpha, beta)))

class Component1DPDF(ComponentPDF):

    def eval( self, features ):
        return self.tf1.Eval(features[0])

    def getEvents( self, n ):
        return np.array(list(self.tf1.GetRandom() for i in range(n))).reshape((-1,1))

class Component2DPDF(ComponentPDF):
    def eval( self, features ):
        return self.tf2.Eval(*features)

    def getEvents( self, n ):
        events = []
        x1=ROOT.Double()
        x2=ROOT.Double()
        for i in range(n):
            self.tf2.GetRandom2(x1,x2)
            events.append( [float(x1),float(x2)] ) 
        return np.array(events).reshape((-1,2))

class Component3DPDF(ComponentPDF):
    def eval( self, features ):
        return self.tf3.Eval(*features)

    def getEvents( self, n ):
        events = []
        x1=ROOT.Double()
        x2=ROOT.Double()
        x3=ROOT.Double()
        for i in range(n):
            self.tf3.GetRandom3(x1,x2,x3)
            events.append( [float(x1),float(x2),float(x3)] ) 
        return np.array(events).reshape((-1,3))

class Exp1D(Component1DPDF):
    ''' normalized exp pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support    = support 
        self.string = "{alpha}/(exp(-{xmin}*{alpha})-exp(-{xmax}*{alpha}))*exp(-(x-{xmin})*({alpha}))".format(alpha=float(parameters[0]), xmin=float(support[0]), xmax = float(support[1]))
        self.tf1 = ROOT.TF1("pdf", self.string, *support)

    @staticmethod
    def _product_norm_factor( alpha, beta, support):
        a = (alpha)/(exp(-support[0]*alpha)-exp(-support[1]*alpha))
        b = (beta)/(exp(-support[0]*beta)-exp(-support[1]*beta))
        aPb = (alpha+beta)/(exp(-support[0]*(alpha+beta))-exp(-support[1]*(alpha+beta)))

        return ( (a*b)/aPb )

    @staticmethod
    def product_norm_factor( alpha, beta, support):
        return reduce(mul, map( lambda ab: Exp1D._product_norm_factor(ab[0],ab[1],support), zip(alpha, beta) ), 1)

class Exp2D(Component2DPDF):
    ''' normalized exp pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support    = support 
        self.string = "{alpha1}/(exp(-{xmin}*{alpha1})-exp(-{xmax}*{alpha1}))*{alpha2}/(exp(-{ymin}*{alpha2})-exp(-{ymax}*{alpha2}))*exp(-(x-{xmin})*({alpha1})-(y-{ymin})*({alpha2}))".format(alpha1=float(parameters[0]), alpha2=float(parameters[1]), xmin=float(support[0]), xmax=float(support[1]), ymin=float(support[2]), ymax=float(support[3]))

        self.tf2 = ROOT.TF2("pdf", self.string, *support)

    @staticmethod
    def product_norm_factor( alpha, beta, support):
        return reduce(mul, map( lambda ab: Exp1D._product_norm_factor(ab[0],ab[1],support), zip(alpha, beta) ), 1)

class Exp3D(Component3DPDF):
    ''' normalized exp pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support    = support 
        self.string = "{alpha1}/(exp(-{xmin}*{alpha1})-exp(-{xmax}*{alpha1}))*{alpha2}/(exp(-{ymin}*{alpha2})-exp(-{ymax}*{alpha2}))*{alpha3}/(exp(-{zmin}*{alpha3})-exp(-{zmax}*{alpha3}))*exp(-(x-{xmin})*({alpha1})-(y-{ymin})*({alpha2})-(z-{zmin})*({alpha3}))".format(alpha1=float(parameters[0]), alpha2=float(parameters[1]), alpha3=float(parameters[2]), xmin=float(support[0]), xmax=float(support[1]), ymin=float(support[2]), ymax=float(support[3]), zmin=float(support[4]), zmax=float(support[5]))

        self.tf3 = ROOT.TF3("pdf", self.string, *support)

    @staticmethod
    def product_norm_factor( alpha, beta, support):
        return reduce(mul, map( lambda ab: Exp1D._product_norm_factor(ab[0],ab[1],support), zip(alpha, beta) ), 1)

class Pow1D(Component1DPDF):
    ''' normalized power-law pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support    = support 
        self.string = "({alpha}-1.)/({xmin}-{xmax}*({xmin}/{xmax})**({alpha}))*(x/({xmin}))**(-{alpha})".format(alpha=float(parameters[0]), xmin=float(support[0]), xmax=float(support[1]))

        self.tf1 = ROOT.TF1("pdf", self.string, *support)

    @staticmethod
    def _product_norm_factor( alpha, beta, support):
        #product_integral = ((-support[0]**(alpha + beta)*support[1] + support[0]*support[1]**(alpha + beta))*(-1 + alpha)*(-1 + beta))/((-support[0]**alpha*support[1] + support[0]*support[1]**alpha)*(-support[0]**beta*support[1] +  support[0]*support[1]**beta)*(-1 + alpha + beta))
        return ((-1. + alpha)*(-1. + beta)*(-(support[0]**(alpha + beta)*support[1]) + support[0]*support[1]**(alpha + beta)))/ ((-1. + alpha + beta)*(-(support[0]**alpha*support[1]) + support[0]*support[1]**alpha)*(-(support[0]**beta*support[1]) + support[0]*support[1]**beta))

    @staticmethod
    def product_norm_factor( alpha, beta, support):
        return reduce(mul, map( lambda ab: Pow1D._product_norm_factor(ab[0], ab[1], support), zip(alpha, beta) ), 1)

class Pow2D(Component2DPDF):
    ''' normalized power-law 2D pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support     = support 
        self.string = "({alpha1}-1.)/({xmin}-{xmax}*({xmin}/{xmax})**({alpha1}))*({alpha2}-1.)/({ymin}-{ymax}*({ymin}/{ymax})**({alpha2}))*(x/({xmin}))**(-{alpha1})*(y/({ymin}))**(-{alpha2})".format(alpha1=float(parameters[0]), alpha2=float(parameters[1]), xmin=float(support[0]), xmax=float(support[1]), ymin=float(support[2]), ymax=float(support[3]))

        self.tf2 = ROOT.TF2("pdf", self.string, *support)

    @staticmethod
    def product_norm_factor( alpha, beta, support):
        return reduce(mul, map( lambda ab: Pow1D._product_norm_factor(ab[0], ab[1], support), zip(alpha, beta) ), 1)

class Pow3D(Component3DPDF):
    ''' normalized power-law 3D pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support     = support 
        self.string = "({alpha1}-1.)/({xmin}-{xmax}*({xmin}/{xmax})**({alpha1}))*({alpha2}-1.)/({ymin}-{ymax}*({ymin}/{ymax})**({alpha2}))*({alpha3}-1.)/({zmin}-{zmax}*({zmin}/{zmax})**({alpha3}))*(x/({xmin}))**(-{alpha1})*(y/({ymin}))**(-{alpha2})*(z/({zmin}))**(-{alpha3})".format(alpha1=float(parameters[0]), alpha2=float(parameters[1]), alpha3=float(parameters[2]), xmin=float(support[0]), xmax=float(support[1]), ymin=float(support[2]), ymax=float(support[3]), zmin=float(support[4]), zmax=float(support[5]))

        self.tf3 = ROOT.TF3("pdf", self.string, *support)

    @staticmethod
    def product_norm_factor( alpha, beta, support):
        return reduce(mul, map( lambda ab: Pow1D._product_norm_factor(ab[0], ab[1], support), zip(alpha, beta) ), 1)

class QuadraticMixturePDF:
    def __init__(self, pdf, parameters, support):

        self.pdf        = pdf
        self.parameters = parameters
        self.support    = support
        self.n_pdf      = len(parameters)

        ## combinatorical factors, combinations, etc. for handling |\theta_i \theta_j p_{i+j}^2|
        self.combinatorical_factors = np.empty(0)
        self.combinations           = np.zeros((0,2), dtype=np.int16) 
        self.prod_to_comb = {}
        for pair in itertools.combinations_with_replacement( range(self.n_pdf), 2 ):
            self.combinatorical_factors = np.append( self.combinatorical_factors, (1 if pair[0]==pair[1] else 2 ) )
            self.combinations           = np.concatenate((self.combinations, np.array([list(pair)])))
        #for combination in self.combinations:
        #    print combination, self.parameters[combination[0]], self.parameters[combination[1]], self.pdf.product_parameters(self.parameters[combination[0]], self.parameters[combination[1]])
        self.combination_pdfs     = [ self.pdf(self.pdf.product_parameters(self.parameters[combination[0]], self.parameters[combination[1]]), self.support) for combination in self.combinations ]
        self.combination_pdfs_pnf = [ self.pdf.product_norm_factor(self.parameters[combination[0]], self.parameters[combination[1]], support) for combination in self.combinations ]
        #for i in range(len(parameters)):
        #    for j in range(len(parameters)):
        #        print parameters[i], parameters[j], self.pdf.product_parameters(parameters[i], parameters[j])
        self.pdfs             = { (i,j): self.pdf(self.pdf.product_parameters(parameters[i], parameters[j]), self.support) for i in range(len(parameters)) for j in range(len(parameters)) }
        self.pdfs_pnf         = { (i,j): self.pdf.product_norm_factor(self.parameters[i], self.parameters[j], support) for i in range(len(parameters)) for j in range(len(parameters)) }

    @staticmethod
    def tilde( theta ):
        #assert len(theta)+1==self.n_pdf, 'Theta vector must have length n_pdf-1'
        return  np.insert(theta, 0, 1, axis=0)
       
    def sigma( self, theta ):
        theta_tilde = self.tilde(theta)
        #print theta_tilde
        #for i in range(len(theta_tilde)):
        #    for j in range(len(theta_tilde)):
        #        print i,j, theta_tilde[i], theta_tilde[j], self.pdfs_pnf[(i,j)]
        #print np.array( [ theta_tilde[i]*theta_tilde[j]*self.pdfs_pnf[(i,j)] for i in range(len(theta_tilde)) for j in range(len(theta_tilde))])
        return np.array( [ theta_tilde[i]*theta_tilde[j]*self.pdfs_pnf[(i,j)] for i in range(len(theta_tilde)) for j in range(len(theta_tilde))]).sum()

    def eval( self, theta, features):
        theta_tilde = self.tilde(theta)
        #print [ (theta_tilde[i], theta_tilde[j], self.pdfs_pnf[(i,j)], self.pdfs[(i,j)].eval(features)) for i in range(len(theta_tilde)) for j in range(len(theta_tilde))]
        #print np.array( [ theta_tilde[i]*theta_tilde[j]*self.pdfs_pnf[(i,j)]*self.pdfs[(i,j)].eval(features) for i in range(len(theta_tilde)) for j in range(len(theta_tilde))])
        return 1./self.sigma(theta)*(np.array( [ theta_tilde[i]*theta_tilde[j]*self.pdfs_pnf[(i,j)]*self.pdfs[(i,j)].eval(features) for i in range(len(theta_tilde)) for j in range(len(theta_tilde))]).sum())
#    def eval_comb( self, theta, features):
#        theta_tilde = self.tilde(theta)
#        #print [ (thet( [ theta_tilde[i]*theta_tilde[j]*self.pdfs_pnf[(i,j)]*self.pdfs[(i,j)].eval(features) for i in range(len(theta_tilde)) for j in range(len(theta_tilde))])
#        return 1./self.sigma(theta)*()

    def getEvents( self, n, theta_ref):

        assert len(theta_ref)+1==len(self.parameters), "Inconsistent theta. I have %i parameters, so I need a vector of length %i. Got %i"%( len(self.parameters),len(self.parameters)-1,len(theta_ref))
        sigma     = self.sigma(theta_ref)
        theta_ref_tilde = self.tilde(theta_ref) 
        fractions = np.array( [ self.combination_pdfs_pnf[i_combination]*self.combinatorical_factors[i_combination]*theta_ref_tilde[combination[0]]*theta_ref_tilde[combination[1]] for i_combination, combination in enumerate(self.combinations) ] )
        fractions /= sigma

        n_for_combination = [ int(n*fractions[i]) for i in range(len(self.combinations)-1) ]
        n_for_combination.append( n - sum(n_for_combination) )

        features = np.concatenate( [ self.combination_pdfs[i_combination].getEvents(n_for_combination[i_combination]) for i_combination in range(len(self.combinations))] )
             
        # shuffling
        indices = np.arange(features.shape[0])
        np.random.shuffle(indices)

        return features[indices]#, weights[indices]

    def getWeights( self, features, theta, theta_ref, only_weights=False):
        assert len(theta)+1==len(self.parameters), "Inconsistent theta. I have %i parameters, so I need a vector of length %i. Got %i"%( len(self.parameters),len(self.parameters)-1,len(theta))
        assert len(theta_ref)+1==len(self.parameters), "Inconsistent theta_ref. I have %i parameters, so I need a vector of length %i. Got %i"%( len(self.parameters),len(self.parameters)-1,len(theta_ref_ref))
        theta_ref_tilde = self.tilde(theta_ref)    
        theta_tilde     = self.tilde(theta)

        comb_factors    = np.array( [ self.combinatorical_factors[i_combination] for i_combination, combination in enumerate(self.combinations) ] )
        probabilities   = np.array( [ [ self.combination_pdfs[i_combination].eval(feature) for i_combination, combination in enumerate(self.combinations) ] for feature in features ] ) 
        theta_tilde_combproducts      = np.array( [  self.combination_pdfs_pnf[i_combination]*theta_tilde[combination[0]]*theta_tilde[combination[1]] for i_combination, combination in enumerate(self.combinations) ] )
        theta_ref_tilde_combproducts  = np.array( [  self.combination_pdfs_pnf[i_combination]*theta_ref_tilde[combination[0]]*theta_ref_tilde[combination[1]] for i_combination, combination in enumerate(self.combinations) ] )

        denominators = np.dot(  comb_factors*probabilities, theta_ref_tilde_combproducts)

        # weights
        numerators   = np.dot(  comb_factors*probabilities, theta_tilde_combproducts)
        weights      = {tuple():numerators/denominators}
        if only_weights:
            return weights[()] 
            #return numerators, denominators 

        # first derivatives
        weights.update( {(i-1,):np.array( [sum([ 2*theta_tilde_j*self.pdfs_pnf[(i,j)]*self.pdfs[(i,j)].eval(feature) for j, theta_tilde_j in enumerate(theta_tilde) ]) for feature in features]) / denominators for i in range(1,len(theta_tilde))} )

        # second derivatives
        weights.update( {(comb[0]-1,comb[1]-1):np.array([2*self.pdfs_pnf[tuple(comb)]*self.pdfs[tuple(comb)].eval(feature) for feature in features])/denominators for i_comb,comb in enumerate(self.combinations) if 0 not in comb} )
        return weights 


if __name__=="__main__":
    support    = [0,5]
    pdf        = Exp1D 
    parameters = [(1.,),(.5,)]
    theta_ref  = [0]
    theta      = [0]

#    support    = [1,5,1,5]
#    pdf        = Exp2D 
#    parameters = [(1,1),(1,1),(1,1)]
#    theta_ref  = [0,0]
#    theta      = [0,0]

#    support    = [1,5,1,5,1,5]
#    pdf        = Pow3D 
#    parameters = [(1.5,1.5,1.5),(1.25,1.25,1.25),(2,2,2)]
#    theta_ref  = [0,0]
#    theta      = [0,0]

    Nevents = 10**4

    mixturePDF = QuadraticMixturePDF( pdf, parameters, support )

    features = mixturePDF.getEvents(Nevents, theta_ref = theta_ref)
    weights  = mixturePDF.getWeights( features, theta = theta, theta_ref = theta_ref)
    #numerators, denominators = mixturePDF.getWeights( features, theta = [-1], theta_ref = theta_ref, only_weights=True)

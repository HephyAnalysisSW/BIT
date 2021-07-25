import ROOT
import numpy as np
from math import exp
import itertools

class ComponentPDF:

    def __init__(self):
        pass

    @staticmethod
    def product_parameters( alpha, beta):

        return tuple(map(sum, zip(alpha, beta)))

class Exp1D(ComponentPDF):
    ''' normalized exp pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support    = support 
        self.string = "{alpha}/(exp(-{x0}*{alpha})-exp(-{xmax}*{alpha}))*exp(-(x-{x0})*({alpha}))".format(alpha=float(parameters[0]), x0=float(support[0]), xmax = float(support[1]))
        self.tf1 = ROOT.TF1("pdf", self.string, *support)

    def eval( self, features ):
        return self.tf1.Eval(features[0])

#    @staticmethod
#    def product_norm_factor( alpha, beta, support):
#        a = (alpha)/(exp(-support[0]*alpha)-exp(-support[1]*alpha))
#        b = (beta)/(exp(-support[0]*beta)-exp(-support[1]*beta))
#        aPb = (alpha+beta)/(exp(-support[0]*(alpha+beta))-exp(-support[1]*(alpha+beta)))
#
#        return (a*b)/aPb

    def getEvents( self, n ):
        return np.array(list(self.tf1.GetRandom() for i in range(n))).reshape((-1,1))

class Exp2D(ComponentPDF):
    ''' normalized exp pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support     = support 
        self.string = "{alpha1}/(exp(-{x0}*{alpha1})-exp(-{xmax}*{alpha1}))*{alpha2}/(exp(-{y0}*{alpha2})-exp(-{ymax}*{alpha2}))*exp(-(x-{x0})*({alpha1})-(y-{y0})*({alpha2})".format(alpha1=float(parameters[0]), alpha2=float(parameters[1]), x0=float(support[0]), xmax=float(support[1]), y0=float(support[2]), ymax=float(support[3]))
        self.tf1 = ROOT.TF2("pdf", self.string, *support)

    def eval( self, features ):
        return self.tf1.Eval(features[0])

#    @staticmethod
#    def product_norm_factor( alpha, beta, support):
#        a = (alpha)/(exp(-support[0]*alpha)-exp(-support[1]*alpha))
#        b = (beta)/(exp(-support[0]*beta)-exp(-support[1]*beta))
#        aPb = (alpha+beta)/(exp(-support[0]*(alpha+beta))-exp(-support[1]*(alpha+beta)))
#
#        return (a*b)/aPb


    def getEvents( self, n ):
        return np.array(list(self.tf1.GetRandom() for i in range(n))).reshape((-1,1))

class Pow1D(ComponentPDF):
    ''' normalized power-law pdf '''
    def __init__( self, parameters, support):
        self.parameters = parameters
        self.support    = support 
        self.string = "({alpha}-1.)/({x0}-{xmax}*({x0}/{xmax})**({alpha}))*(x/({x0}))**(-{alpha})".format(alpha=float(parameters[0]), x0=float(support[0]), xmax=float(support[1]))

        self.tf1 = ROOT.TF1("pdf", self.string, *support)
    
    def eval( self, features ):
        return self.tf1.Eval(features[0])

#    @staticmethod
#    def product_norm_factor( self, alpha, beta):
#        return ((-support[0]**(alpha + beta)*support[1] + support[0]*support[1]**(alpha + beta))*(-1 + alpha)*(-1 + beta))/((-support[0]**alpha*support[1] + support[0]*support[1]**alpha)*(-support[0]**beta*support[1] +  support[0]*support[1]**beta)*(-1 + alpha + beta))

    def getEvents( self, n ):
        return np.array(list(self.tf1.GetRandom() for i in range(n))).reshape((-1,1))

class Quadratic1DMixturePDF:
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
        self.combination_pdfs = [ self.pdf(self.pdf.product_parameters(self.parameters[combination[0]], self.parameters[combination[1]]), self.support) for combination in self.combinations]
        #for i in range(len(parameters)):
        #    for j in range(len(parameters)):
        #        print parameters[i], parameters[j], self.pdf.product_parameters(parameters[i], parameters[j])
        self.pdfs             = { (i,j): self.pdf(self.pdf.product_parameters(parameters[i], parameters[j]), self.support) for i in range(len(parameters)) for j in range(len(parameters)) }

    @staticmethod
    def tilde( theta ):
        #assert len(theta)+1==self.n_pdf, 'Theta vector must have length n_pdf-1'
        return  np.insert(theta, 0, 1, axis=0)
       
    def sigma( self, theta ):
        theta_tilde = self.tilde(theta)
        return np.array( [ theta_tilde[i]*theta_tilde[j] for i in range(len(theta_tilde)) for j in range(len(theta_tilde))]).sum()

    def getEvents( self, n, theta_ref):

        assert len(theta_ref)+1==len(self.parameters), "Inconsistent theta. I have %i parameters, so I need a vector of length %i. Got %i"%( len(self.parameters),len(self.parameters)-1,len(theta_ref))
        sigma     = self.sigma(theta_ref)
        theta_ref_tilde = self.tilde(theta_ref) 
        fractions = np.array( [ self.combinatorical_factors[i_combination]*theta_ref_tilde[combination[0]]*theta_ref_tilde[combination[1]] for i_combination, combination in enumerate(self.combinations) ] )
        fractions /= sigma

        n_for_combination = [ int(n*fractions[i]) for i in range(len(self.combinations)-1) ]
        n_for_combination.append( n - sum(n_for_combination) )

        #for i_combination, combination in enumerate(self.combinations):
            #print ("%i events from comb %i %r with parameter %f" % (n_for_combination[i_combination], i_combination, combination, self.pdf.product_parameters(self.parameters[combination[0]], self.parameters[combination[1]])))

        features = np.concatenate( [ self.combination_pdfs[i_combination].getEvents(n_for_combination[i_combination]) for i_combination in range(len(self.combinations))] )
             
        # shuffling
        indices = np.arange(features.shape[0])
        np.random.shuffle(indices)

        return features[indices]#, weights[indices]

    def getWeights( self, features, theta, theta_ref):
        assert len(theta)+1==len(self.parameters), "Inconsistent theta. I have %i parameters, so I need a vector of length %i. Got %i"%( len(self.parameters),len(self.parameters)-1,len(theta))
        assert len(theta_ref)+1==len(self.parameters), "Inconsistent theta_ref. I have %i parameters, so I need a vector of length %i. Got %i"%( len(self.parameters),len(self.parameters)-1,len(theta_ref_ref))
        theta_ref_tilde = self.tilde(theta_ref)    
        theta_tilde     = self.tilde(theta)

        comb_factors    = np.array( [ self.combinatorical_factors[i_combination] for i_combination, combination in enumerate(self.combinations) ] )
        probabilities   = np.array( [ [ self.combination_pdfs[i_combination].eval(feature) for i_combination, combination in enumerate(self.combinations) ] for feature in features ] ) 
        theta_tilde_combproducts      = np.array( [ theta_tilde[combination[0]]*theta_tilde[combination[1]] for i_combination, combination in enumerate(self.combinations) ] )
        theta_ref_tilde_combproducts  = np.array( [ theta_ref_tilde[combination[0]]*theta_ref_tilde[combination[1]] for i_combination, combination in enumerate(self.combinations) ] )

        denominators = np.dot(  comb_factors*probabilities, theta_ref_tilde_combproducts)

        # weights
        numerators   = np.dot(  comb_factors*probabilities, theta_tilde_combproducts)
        weights      = {tuple():numerators/denominators} 

        # first derivatives
        weights.update( {(i-1,):np.array( [sum([ 2*theta_tilde_j*self.pdfs[(i,j)].eval(feature) for j, theta_tilde_j in enumerate(theta_tilde) ]) for feature in features]) / denominators for i in range(1,len(theta_tilde))} )

        # second derivatives
        weights.update( {(comb[0]-1,comb[1]-1):np.array([2*self.pdfs[tuple(comb)].eval(feature) for feature in features])/denominators for i_comb,comb in enumerate(self.combinations) if 0 not in comb} )
        return weights 


if __name__=="__main__":
    support    = [1,5]
    pdf        = Pow1D 
    parameters = [(1,),(1,),(1,)]
    theta_ref  = [0,0]
    mixturePDF = Quadratic1DMixturePDF( pdf, parameters, support )

    features = mixturePDF.getEvents(20, theta_ref = theta_ref)

    theta      = [0,0]
    weights  = mixturePDF.getWeights( features, theta = theta, theta_ref = theta_ref)

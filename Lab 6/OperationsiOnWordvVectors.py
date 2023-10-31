import numpy as np
from w2v_utils import *



words, word_to_vec_map = read_glove_vecs('../../readonly/glove.6B.50d.txt')



class VectorOperation:
    def cosine_similarity_test():
        father = word_to_vec_map["father"]
        mother = word_to_vec_map["mother"]
        ball = word_to_vec_map["ball"]
        crocodile = word_to_vec_map["crocodile"]
        france = word_to_vec_map["france"]
        italy = word_to_vec_map["italy"]
        paris = word_to_vec_map["paris"]
        rome = word_to_vec_map["rome"]

        print("cosine_similarity(father, mother) = ", VectorOperation.cosine_similarity(father, mother))
        print("cosine_similarity(ball, crocodile) = ",VectorOperation.cosine_similarity(ball, crocodile))
        print("cosine_similarity(france - paris, rome - italy) = ",VectorOperation.cosine_similarity(france - paris, rome - italy))

    def cosine_similarity(u, v):
        """
        Cosine similarity reflects the degree of similarity between u and v
            
        Arguments:
            u -- a word vector of shape (n,)          
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
        """
        
        distance = 0.0
        
        # Compute the dot product between u and v (≈1 line)
        dot = np.dot(u,v)
        # Compute the L2 norm of u (≈1 line)
        norm_u = np.sqrt(np.sum(u**2))
        
        # Compute the L2 norm of v (≈1 line)
        norm_v = np.sqrt(np.sum(v**2))
        # Compute the cosine similarity defined by formula (1) (≈1 line)
        cosine_similarity = dot/(norm_u*norm_v)
        
        return cosine_similarity
    
    def complete_analogy_test():
        triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
        for triad in triads_to_try:
            print ('{} -> {} :: {} -> {}'.format( *triad, VectorOperation.complete_analogy(*triad,word_to_vec_map)))

    def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
        """
        Performs the word analogy task as explained above: a is to b as c is to ____. 
        
        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
        
        Returns:
        best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
        """
        
        # convert words to lowercase
        word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
        
        ### START CODE HERE ###
        # Get the word embeddings e_a, e_b and e_c (≈1-3 lines)
        e_a = word_to_vec_map.get(word_a)
        e_b = word_to_vec_map.get(word_b)
        e_c = word_to_vec_map.get(word_c)
        ### END CODE HERE ###
        
        words = word_to_vec_map.keys()
        max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
        best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

        # to avoid best_word being one of the input words, skip the input words
        # place the input words in a set for faster searching than a list
        # We will re-use this set of input words inside the for-loop
        input_words_set = set([word_a, word_b, word_c])
        
        # loop over the whole word vector set
        for w in words:        
            # to avoid best_word being one of the input words, skip the input words
            if w in input_words_set:
                continue
            
            # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
            cosine_sim = VectorOperation.cosine_similarity(np.subtract(e_b,e_a), np.subtract(word_to_vec_map.get(w),e_c))
            
            # If the cosine_sim is more than the max_cosine_sim seen so far,
                # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w
            
        return best_word
        
    def neutralize_test():
        e = "receptionist"
        print("cosine similarity between " + e + " and g, before neutralizing: ", VectorOperation.cosine_similarity(word_to_vec_map["receptionist"], g))

        e_debiased = VectorOperation.neutralize("receptionist", g, word_to_vec_map)
        print("cosine similarity between " + e + " and g, after neutralizing: ", VectorOperation.cosine_similarity(e_debiased, g))

    def neutralize(word, g, word_to_vec_map):
        """
        Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
        This function ensures that gender neutral words are zero in the gender subspace.
        
        Arguments:
            word -- string indicating the word to debias
            g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
            word_to_vec_map -- dictionary mapping words to their corresponding vectors.
        
        Returns:
            e_debiased -- neutralized word vector representation of the input "word"
        """
        
        # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
        e = word_to_vec_map[word]
        
        # Compute e_biascomponent using the formula given above. (≈ 1 line)
        e_biascomponent = (np.dot(e, g)/np.linalg.norm(g)**2)*g
    
        # Neutralize e by subtracting e_biascomponent from it 
        # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
        e_debiased = e - e_biascomponent
        
        return e_debiased
    
    def equalize_test():
        print("cosine similarities before equalizing:")
        print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", VectorOperation.cosine_similarity(word_to_vec_map["man"], g))
        print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", VectorOperation.cosine_similarity(word_to_vec_map["woman"], g))
        print()
        e1, e2 = VectorOperation.equalize(("man", "woman"), g, word_to_vec_map)
        print("cosine similarities after equalizing:")
        print("cosine_similarity(e1, gender) = ", VectorOperation.cosine_similarity(e1, g))
        print("cosine_similarity(e2, gender) = ", VectorOperation.cosine_similarity(e2, g))

    def equalize(pair, bias_axis, word_to_vec_map):
        """
        Debias gender specific words by following the equalize method described in the figure above.
        
        Arguments:
        pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
        bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
        word_to_vec_map -- dictionary mapping words to their corresponding vectors
        
        Returns
        e_1 -- word vector corresponding to the first word
        e_2 -- word vector corresponding to the second word
        """
        
        # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
        w1, w2 = pair[0], pair[1]
        e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
        
        # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
        mu = (e_w1 + e_w2)/2

        # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
        mu_B = (np.dot(mu, bias_axis)/np.linalg.norm(bias_axis)**2)*bias_axis
        mu_orth = mu - mu_B

        # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
        e_w1B = (np.dot(e_w1, bias_axis)*np.linalg.norm(bias_axis)**2)*bias_axis
        e_w2B = (np.dot(e_w2, bias_axis)*np.linalg.norm(bias_axis)**2)*bias_axis
            
        # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
        corrected_e_w1B = np.sqrt(1 - np.linalg.norm(mu_orth)**2)*((e_w1B - mu_B)/np.abs((e_w1 - mu_orth) - mu_B))
        corrected_e_w2B = np.sqrt(1 - np.linalg.norm(mu_orth)**2)*((e_w2B - mu_B)/np.abs((e_w2 - mu_orth) - mu_B))

        # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
        e1 = corrected_e_w1B + mu_orth
        e2 = corrected_e_w2B + mu_orth
                                                                    
        
        return e1, e2

def main():
    pass

if __name__ == '__main__':
    main()
    
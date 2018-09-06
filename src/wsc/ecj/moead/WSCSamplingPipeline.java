package wsc.ecj.moead;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.primitives.Doubles;

import ec.BreedingPipeline;
import ec.EvolutionState;
import ec.Individual;
import ec.util.Parameter;
import wsc.data.pool.Service;
import wsc.ecj.nsga2.SequenceVectorIndividual;
import wsc.nhbsa.NHBSA;
import wsc.problem.WSCInitializer;

import ec.Subpopulation;
import ec.multiobjective.MultiObjectiveFitness;

public class WSCSamplingPipeline extends BreedingPipeline {

	private static final long serialVersionUID = 1L;

	@Override
	public Parameter defaultBase() {
		return new Parameter("wscsamplingpipeline");
	}

	@Override
	public int numSources() {
		return 1;
	}

	public int produce(int min, int max, int start, int subpopulation, Individual[] inds, EvolutionState state,
			int thread) {

		WSCInitializer init = (WSCInitializer) state.initializer;

		// obtain the subproblem representative

		SequenceVectorIndividual bestNeighbour = (SequenceVectorIndividual) inds[start].clone();

		double bestScore = init.calculateTchebycheffScore(bestNeighbour, start);
		bestNeighbour.setTchebycheffScore(bestScore);

		if (WSCInitializer.pop_updated != null) {
			WSCInitializer.pop_updated.clear();
		}

		// learn NHM from subproblem, but it is penalized on based the tchebycheff score
		WSCInitializer.pop_updated = sampleNeighbors(init, start, state);

		for (int i = 0; i < WSCInitializer.pop_updated.size(); i++) {
			SequenceVectorIndividual neighbour = (SequenceVectorIndividual) inds[start].clone();
			updatedIndi(neighbour.genome, WSCInitializer.pop_updated.get(i));

			// Calculate fitness of neighbor
			neighbour.calculateSequenceFitness(neighbour, init, state);

			// Calculate tchebycheffScore
			double score = init.calculateTchebycheffScore(neighbour, start);
			bestNeighbour.setTchebycheffScore(score);
			if (score < bestScore) {
				bestScore = score;
				bestNeighbour = neighbour;
			}

		}

		inds[start] = bestNeighbour;
		inds[start].evaluated = false;

		return 1;
	}

	private List<int[]> sampleNeighbors(WSCInitializer init, int start, EvolutionState state) {
		// Get population
		Subpopulation pop = state.population.subpops[0];
		// System.out.println("learn a NHM from a pop size: " + pop.individuals.length);
		NHBSA nhbsa = new NHBSA(WSCInitializer.numNeighbours, WSCInitializer.dimension_size);

		int[][] m_generation = new int[WSCInitializer.numNeighbours][WSCInitializer.dimension_size];

		double consinSimilarity[] = new double[WSCInitializer.numNeighbours];

		for (int m = 0; m < WSCInitializer.numNeighbours; m++) {

			MultiObjectiveFitness fit = (MultiObjectiveFitness) pop.individuals[m].fitness;
			double[] array = { fit.getObjective(0), fit.getObjective(1) };
			consinSimilarity[m] = cosineSimilarity(array, init.weights[start]);

			for (int n = 0; n < WSCInitializer.dimension_size; n++) {
				m_generation[m][n] = ((SequenceVectorIndividual) (pop.individuals[m])).serQueue.get(n);
			}
		}

		nhbsa.setM_pop(m_generation);
		nhbsa.setM_L(WSCInitializer.dimension_size);
		nhbsa.setM_N(WSCInitializer.numNeighbours);
		nhbsa.setNormalizedConsineSIM(consinSimilarity);

		// Sample numLocalSearchTries number of neighbors
		return nhbsa.sampling4NHBSA(WSCInitializer.numLocalSearchTries, WSCInitializer.random);
	}

	public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
		double dotProduct = 0.0;
		double normA = 0.0;
		double normB = 0.0;
		for (int i = 0; i < vectorA.length; i++) {
			dotProduct += vectorA[i] * vectorB[i];
			normA += Math.pow(vectorA[i], 2);
			normB += Math.pow(vectorB[i], 2);
		}
		return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}

	private void updatedIndi(Service[] genome, int[] updateIndi) {
		for (int n = 0; n < updateIndi.length; n++) {
			genome[n] = WSCInitializer.Index2ServiceMap.get(updateIndi[n]);
		}
	}

}

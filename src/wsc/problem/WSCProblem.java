package wsc.problem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ec.EvolutionState;
import ec.Individual;
import ec.Problem;
import ec.Subpopulation;
import ec.multiobjective.MultiObjectiveFitness;
import ec.simple.SimpleProblemForm;
import wsc.data.pool.InitialWSCPool;
import wsc.data.pool.Service;
import wsc.ecj.nsga2.SequenceVectorIndividual;
import wsc.graph.ServiceGraph;

public class WSCProblem extends Problem implements SimpleProblemForm {
	private static final long serialVersionUID = 1L;

	@Override
	public void evaluate(final EvolutionState state, final Individual ind, final int subpopulation,
			final int threadnum) {
		if (ind.evaluated)
			return;

		if (!(ind instanceof SequenceVectorIndividual))
			state.output.fatal("Whoa!  It's not a SequenceVectorIndividual!!!", null);

		SequenceVectorIndividual individual = (SequenceVectorIndividual) ind;
		List<Integer> fullSerQueue = new ArrayList<Integer>();
		List<Integer> usedSerQueue = new ArrayList<Integer>();

		WSCInitializer init = (WSCInitializer) state.initializer;

		// use ind2 to generate graph
		InitialWSCPool.getServiceCandidates().clear();
		List<Service> serviceCandidates = new ArrayList<Service>(Arrays.asList(individual.genome));
		InitialWSCPool.setServiceCandidates(serviceCandidates);

		ServiceGraph graph = init.graGenerator.generateGraph(fullSerQueue);
		// create a queue of services according to breath first search
		List<Integer> usedQueue = init.graGenerator.usedQueueofLayers("startNode", graph, usedSerQueue);
		// set the position of the split position of the queue
		individual.setSplitPosition(usedQueue.size()); // index from 0 to (splitposition-1)

		// add unused queue to form a complete a vector-based individual
		List<Integer> serQueue = init.graGenerator.completeSerQueueIndi(usedQueue, fullSerQueue);
		// Set serQueue to individual(do I need deep clone ?)
		individual.serQueue.addAll(serQueue);

		individual.setStrRepresentation(graph.toString());
		// evaluate updated updated_graph
		init.eval.aggregationAttribute(individual, graph);

		((MultiObjectiveFitness) individual.fitness).setObjectives(state, init.eval.calculateFitness(individual));
		individual.evaluated = true;
	}

}
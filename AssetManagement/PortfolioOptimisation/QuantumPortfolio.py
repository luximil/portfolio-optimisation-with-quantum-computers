# ----------------------------------------------------------------------------
# Written by Francisco Jose Manjon Cabeza Garcia for his Master Thesis at    |
# Technische Universität Berlin.                                             |
#                                                                            |
# This file contains the QuantumPortfolio class, which implements the        |
# quantum (discrete) portfolio optimisation methods based on Markowitz'      |
# portfolio optimisation model.                                              |
# ----------------------------------------------------------------------------

from .ClassicalPortfolio import ClassicalPortfolio
import pandas as pd
from pennylane import qaoa
import pennylane as qml
from pennylane import numpy as npp
import time


class QuantumPortfolio(ClassicalPortfolio):

    def GetPortfolioOptimisationCostHamiltonian(
            self,
            asset_manager_param: float,
            net_positions: int,
            alpha: float,
            beta: float,
            cov_matrix: pd.DataFrame,
            expected_returns: pd.Series,
            transaction_costs: pd.Series,
            initial_portfolio: pd.Series,
            test_runs: int = 20,
            test_tol: float = 10e-8,
            simplify: bool = False) -> qml.Hamiltonian:
        """"Compute the cost Hamiltonian to encode the following binary
        optimisation model:
        f(z+, z-) := asset_manager_lambda * risk_contribution -
                     (1-asset_manager_lambda) * (gross_returns - tc_contrib) +
                     net_pos_const + deg_const,
        where

        risk_contribution = 1/net_positions * w^T * cov_matrix * w,

        gross_returns = exp_returns^T * w,

        tc_contrib = transaction_costs^T * abs(w - initial_portfolio),

        net_pos_const = alpha * (count - net_positions)^2

        deg_const = beta * (z+^T z-),

        and count is equal to the number of entries in the portfolio which
        are not zero., and w := 1/net_positions * (z+ - z-) is a vector with
        elements in {-1/net_positions, 0, 1/net_positions}.

        Parameters
        ----------
        asset_manager_param : float
            Expresses the trade-off between risk and return. Setting it to 0
            maximises return only. Setting it to 1 minimises volatility only.
        net_positions : int
            Only portfolios with exactly this number of positions, i.e.,
            non-zero elements, will be considered feasible solutions in the
            soft-constraint.
        alpha : float
            Penalty parameter for the cardinality soft-constraint.
        beta : float
            Penalty parameters for the degenerate solutions infeasibility
            soft-constraint.
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns.
        expected_returns : pandas.Series
            Expected returns indexed vector.
        transaction_costs : pandas.Series
            Indexed vector containing the transaction costs of each asset in
            percentage terms.
        initial_portfolio : pandas.Series
            Indexed vector containing the initial portfolio asset weights,
            which will be considered in the computation of transaction costs
            when rebalancing the portfolio from initial_portfolio to optimal
            portfolio.
        test_runs : int, default 20
            Number of random binary vectors of size 2*N used to test that each
            QUBO matrix used to construct the final Q matrix were correctly
            constructed.
        test_tol : float, default 10e-8
            Maximum deviations from the actual value allowed in the testing of
            each QUBO matrix used to construct the final Q matrix were
            correctly constructed.
        qs_export_path : str, default None
            If not None, the resulting Q matrix is exported in .qs integer
            format to this path.
        export_decimal_places : int, default 4
            Number of decimal places to round the entries in Q before export.
        simplify : bool, default True
            If True, the computed Hamiltonian is simplified using the .simplfy
            method from Pennylane. This can help reduce the number of elements
            needed to define the Hamiltonian. If False, no simplify method is
            applied to the computed Hamiltonian.

        Returns
        -------
        total_cost_h : pennylane.Hamiltonian
            pennylane.Hamiltonian object which encondes the discrete equal
            weight Markowitz portfolio optimisation model. The wires of the
            pennylane.Hamiltonian object are defined by the covariance matrix
            row and column index.
        """

        # Get the number of assets in the portfolio.
        N = self.NumberOfAssetsInPortfolio

        assert cov_matrix.shape == (N, N), \
            (f"The covariance matrix was expected to have shape ({N}, {N}),"
             f" but it has shape {cov_matrix.shape}.")

        assert expected_returns.shape[0] == N, \
            ("The dimension of vector of expected returns does not match the"
             " number of assets in the portfolio.")

        assert transaction_costs.shape[0] == N, \
            ("The dimension of vector of transaction costs does not match the"
             " number of assets in the portfolio.")

        assert initial_portfolio.shape[0] == N, \
            ("The dimension of initial portfolio vector does not match the"
             " number of assets in the portfolio.")

        # Get QUBO matrix which will be enconded into the cost Hamiltonian.
        Q = self.GetDiscreteEqualWeightMarkowitzPortfolioQUBOMatrix(
            asset_manager_param=asset_manager_param,
            net_positions=net_positions,
            alpha=alpha,
            beta=beta,
            cov_matrix=cov_matrix,
            expected_returns=expected_returns,
            transaction_costs=transaction_costs,
            initial_portfolio=initial_portfolio,
            test_runs=test_runs,
            test_tol=test_tol)

        cost_obs = [qml.Identity(Q.index[0])]
        cost_coeffs = [0]

        for k in range(Q.shape[0]):
            # Add diagonal elements to the coefficient of the identity
            # observable.
            cost_coeffs[0] = cost_coeffs[0] + Q.iloc[k, k] / 2

            # Define the coefficient of the k-th Pauli-Z operator.
            pauliZ_k_coeff = -npp.sum(Q.iloc[k, :]) / 2

            for l in range(k+1, Q.shape[1]):
                # Add upper triangular elements to the coefficient of the
                # identity observable.
                cost_coeffs[0] = (cost_coeffs[0]
                                  + (Q.iloc[k, l] + Q.iloc[l, k]) / 4)

                cost_coeffs.append((Q.iloc[k, l] + Q.iloc[l, k]) / 4)
                cost_obs.append(qml.PauliZ(Q.index[k])
                                @ qml.PauliZ(Q.columns[l]))

            # Rescale the coefficient of the k-th Pauli-Z operator, add it to
            # the list of coefficients.
            cost_coeffs.append(pauliZ_k_coeff)
            # Add the k-th Pauli-Z operator to the lsit of observables.
            cost_obs.append(qml.PauliZ(Q.index[k]))

        assert len(cost_obs) == len(cost_coeffs), \
            ("The number of obsevations and coefficients for the objective"
             " value function do not match.")

        cost_h = qml.Hamiltonian(coeffs=npp.array(cost_coeffs),
                                 observables=cost_obs)
        if simplify:
            cost_h = cost_h.simplify()

        return cost_h

    def GetPortfolioOptimisationMixerHamiltonian(
            self,
            wires: qml.wires.Wires,
            approach: str = "PauliX") -> qml.Hamiltonian:
        """Compute and return a mixer Hamiltonian to later use in a QAOA
        circuit.

        Parameters
        ----------
        wires : pennylane.wires.Wires
            Wires (qubits) to the define the mixer Hamiltonian.
        approach : str, default "PauliX"
            Approach to use for the mixer Hamiltonian definition. The
            following approaches are currently implemented:
            - "PauliX" (default): compute the mixer Hamiltonian as the sum
              of PauliX operators acting on each of the wires (qubits).
            - "LongShortPauliXY": compute the mixer Hamiltonian as the sum of
              1/2*(PauliX(long)@PauliX(short) + PauliY(long)@PauliY(short))
              for each long and short asset wire.

        Returns
        -------
        mixer_h : pennylane.Hamiltonian
            Mixer Hamiltonian for the QAOA circuit.
        """

        if approach == "PauliX":
            coeffs = [1 for wires in wires]
            obs = [qml.PauliX(wires) for wires in wires]
            mixer_h = qml.Hamiltonian(coeffs=coeffs, observables=obs)

            return mixer_h

        if approach == "LongShortPauliXY":
            # Define an empty list where the summands for each asset are saved.
            mixer_h_summands = []

            wires = sorted(wires)
            i = 0
            while i < len(wires):
                # Get asset long wire.
                long_wire = wires[i]
                # Get asset short wire.
                short_wire = wires[i+1]

                assert long_wire.endswith("long"), \
                    ("The long wire variable identified does not contain the"
                     " suffix \"long\".")
                assert short_wire.endswith("short"), \
                    ("The short wire variable identified does not contain the"
                     " suffix \"short\".")
                assert long_wire.split("_")[0] == short_wire.split("_")[0], \
                    ("The long and short wire variables identified do not"
                     " represent the same asset.")

                # Get the PauliXY mixer component linking the long and short
                # qubits for the selected asset.
                asset_mixer_element = (
                    qml.PauliX(long_wire) @ qml.PauliX(short_wire)
                    + qml.PauliY(long_wire) @ qml.PauliY(short_wire)
                    )
                # Add the computed component to the list of mixer Hamiltonian
                # summands.
                mixer_h_summands.append(asset_mixer_element)

                # Skip to the next asset wires.
                i = i+2

            # Compute the mixer Hamiltonian as the sum of the summands just
            # computed for each long-short wire pairs times 1/2.
            mixer_h = 1/2 * sum(mixer_h_summands)

            return mixer_h

        else:
            raise NotImplementedError("The approach not implemented.")

    def GetPortfolioOptimisationQAOACircuit(
            self,
            hamiltonians: list[tuple[qml.Hamiltonian, str]],
            eval_cost_h: qml.Hamiltonian,
            depth: int,
            quantum_device_name: str = "default.qubit",
            differentiation_method="parameter-shift",
            init_state: npp.ndarray | None = None):
        """Construct a QAOA circuit using a given cost Hamiltonian, mixer
        Hamiltonian and depth.

        Parameters
        ----------
        hamiltonians : list[tuple[pennylane.Hamiltonian, str]]
            List containing tuples with a Hamiltonian and a string indicating
            if the Hamiltonian is a cost ("cost") or a mixer ("mixer")
            Hamiltonian to define a layer in the QAOA circuit. The order of
            the list is considered in the definition of the layer. Although
            QAOA circuits are defined as circuits with a cost plus a mixer
            layer which is repeated, this parameter allows to define QAOA
            circuit layers which can consist of any combination of cost and
            mixer layers.
        eval_cost_h : pennylane.Hamiltonian
            Hamiltonian used to evaluate the quantum circuit's expected cost
            value. Usually, this is the same as the cost Hamiltonian in the
            hamiltonians list parameter if only one cost Hamiltonian and one
            mixer Hamiltionians are passed, as the QAOA circuit is normally
            defined.
        depth : int
            Number of cost+mixer layers to construct the circuit.
        quantum_device_name : str, default "default.qubit"
            Name of the quantum device on which the circuit is run. The
            default device used is PennyLane's default.qubit device. The list
            of available devices can be found here:
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
        differentiation_method : str, default "parameter-shift"
            Differentiation method used to compute the gradients of the
            cost_function returned by this method. The list of differentiation
            methods available in Pennylane can be found here:
            https://docs.pennylane.ai/en/stable/introduction/interfaces.html#gradients.
        init_state : numpy.ndarray, default None
            Vector with the intial quantum state before applying the circuit.
            This vector contains the probabilities of observing each possible
            bit-string, where i-th bit-string is equal to the binary
            representation of the integer i. Hence, the 2-norm of this vector
            must be equal to 1. If None, Apply an initial layer consisting of
            one Hadamard gate to each qubit to bring them into a superposition
            state.

        Returns
        -------
        circuit
            QAOA quantum circuit.
        cost_function : pennylane.node
            Function which takes the circuit parameters as input, and returns
            the expected values of the observable, i.e., the cost Hamiltonian
            given in parameter eval_cost_h. It can later be used to optimise
            these circuit parameters.
        circuit_probabilities : pennylane.node
            Function which takes the circuit parameters as input, and returns
            the probability distribution of all possible qubit state strings.
        """

        # Get the number of assets.
        N = self.NumberOfAssetsInPortfolio

        assert len(eval_cost_h.wires) == 2*N, \
            ("The cost Hamiltonian used to evaluate the quantum circuit's"
             f" expected value passed acts on {len(eval_cost_h.wires)} wires,"
             f" while {2*N} were expected (twice the number of assets).")

        for i, (ham, ham_type) in enumerate(hamiltonians):
            assert ham.wires == eval_cost_h.wires, \
                (f"The Hamiltonian passed at index {i} acts on wires"
                 f" {ham.wires}, which are not the same as the circuit"
                 f" evaluation's Hamiltonian wires {eval_cost_h.wires}.")

            assert ham_type == "cost" or ham_type == "mixer", \
                (f"The Hamiltonian passed at index {i} has an invalid type."
                 " Only cost and mixer types are allowed.")

        assert depth >= 1, "The selected depth is strictly smaller than 1."

        # Define the wires to later initialise the circuit with the correct
        # number of qubits.
        wires = eval_cost_h.wires

        # Define the number of shots to use for expected value computations as
        # the default vlaue 1000 explicitely. But only if the differentiation
        # method is not "adjoint", because for it expected values are computed
        # exactly.
        exp_val_shots = 1000
        if differentiation_method == "adjoint":
            exp_val_shots = None
        # Define the quantum device on which the quantum circuit will be run.
        dev = qml.device(name=quantum_device_name,
                         wires=wires,
                         shots=exp_val_shots)

        # Define a layer of the QAOA quantum circuit, which consists of the
        # layers defined in the hamiltonians list parameter.
        def qaoa_layer(*argv):
            for i, (ham, ham_type) in enumerate(hamiltonians):
                if ham_type == "cost":
                    qaoa.cost_layer(hamiltonian=ham, gamma=argv[i])
                elif ham_type == "mixer":
                    qaoa.mixer_layer(hamiltonian=ham, alpha=argv[i])

        # Define a method for the quantum circuit using the qaoa_layer method.
        def circuit(params, **kwargs):
            if init_state is None:
                # Apply an initial layer consisting of one Hadamard gate to
                # each qubit to bring them into a superposition state.
                for w in wires:
                    qml.Hadamard(wires=w)
            else:
                assert init_state.shape[0] == 2**len(wires), \
                    ("The length of the  initial quantum state does not match"
                     " the number of the wires (qubits) in the circuit.")

                # Set the initial qubit state to the one given by init_state.
                qml.QubitStateVector(state=init_state, wires=wires)

            # After the Hadamard gates layer, add the QAOA layers (as many as
            # depth parameter) with their parameters to optimise later.
            qml.layer(qaoa_layer, depth, *params)

        @qml.qnode(device=dev, diff_method=differentiation_method)
        def cost_function(params):
            circuit(params)
            return qml.expval(op=eval_cost_h)

        @qml.qnode(dev)
        def circuit_probabilities(params):
            circuit(params)
            return qml.probs(wires=wires)

        return circuit, cost_function, circuit_probabilities

    def TrainQAOACircuitParameters(self,
                                   cost_function: qml.qnode,
                                   step_eps: float,
                                   step_reduction_factor: float,
                                   min_step_size: float,
                                   step_reduction_patience: int,
                                   max_patience: int,
                                   max_iterations: int,
                                   optimiser,
                                   initial_params: npp.ndarray,
                                   return_training_progress: bool = False,
                                   verbose: bool = False) -> tuple[npp.ndarray,
                                                                   float,
                                                                   dict]:
        """This method optimises the parameters of the QAOA circuit defined in
        the cost_function parameter.

        Parameters
        ----------
        cost_function : pennylane.node
            QAOA circuit cost function to optimise.
        step_eps : float
            Minimum cost function value decrease to consider the current step
            an improvement.
        step_reduction_factor : float
            If the optimiser gets stuck and cannot find a better solution than
            the current optimal one, reduce the step size by this factor.
        min_step_size : float
            Minimum step size for the optimiser. If the step size is strictly
            less than this parameter after reducing it, the optimisation is
            stopped.
        step_reduction_patience : int
            If not None, maximum number of iterations without improvement
            before reducing the step size of the optimiser.
            If None, then no optimiser step reduction will ever be applied.
        max_patience : int
            If not None, maximum number of iterations without improvement to
            stop the optimisation.
            If None, then no fixed limit to the number of iterations without
            improvement is applied.
        max_iterations : int
            If not None, maximum number of total iterations. If None, then no
            fixed limit to the number of iterations is applied.
        optimiser
            PennyLane optimiser to use.
        initial_params : numpy.ndarray
            Initial circuit parameters.
        return_training_progress: bool, default False
            If True, the objective function value and it optimal value up to
            then are saved on each iteration and returned at the end.
        verbose : bool, default False
            If True, print after each optimisation iteration the lowest
            objective function value found up to then.

        Returns
        -------
        opt_params : numpy.ndarray
            Parameters with the lowest objective function value found.
        opt_cost : float
            Lowest objective function value found, i.e., objective function
            value for opt_params.
        iters_log : dict of numpy.ndarray # TODO: adjust to use pd.DataFrame.
            Dictionary with the following keys:
            - "Gradient Descent Cost": Timeseries with the objective function
              value at each optimisation iteration,
            - "Optimal Cost": Timeseries with the lowest objective function
              value up to each optimisation iteration,
            - "Iteration Running Time": Timeseries with the running time in
              seconds of each optimisation iteration.
        """

        if step_reduction_patience is None:
            step_reduction_patience = npp.inf

        if max_patience is None:
            max_patience = npp.inf

        if max_iterations is None:
            max_iterations = npp.inf

        assert step_reduction_patience > 0, \
            ("The step reduction patience parameter given must be strictly"
             " greater than 0.")
        assert max_patience > 0, \
            ("The maximum patience parameter given must be strictly"
             " greater than zero.")
        assert max_iterations > 0, \
            ("The maximum iterations parameter given must be strictly"
             " greater than zero.")

        params = initial_params

        # Define variables to keep track of the optimal parameters and their
        # optimal cost during the optimisation.
        opt_params = params
        opt_cost = cost_function(params)

        if return_training_progress:
            # Define arrays to save the cost and optimal cost at each
            # optimisation iteration to later plot the results over
            # time or optimisation iteration count.
            iters_log = {
                "Gradient Descent Cost": [opt_cost],
                "Optimal Cost": [opt_cost],
                "Iteration Running Time": [0]
                }
        else:
            iters_log = {}

        # Define a variable to count the total number of iterations.
        iterations = 0
        # Define a counter for the number of iterations without improvement.
        iters_without_improvement = 0

        if verbose:
            print(f"Initial expected value of the cicuit {opt_cost}.")

        # While the maximum number of iterations without improvement or the
        # fixed maximum number of iterations has not been reached,
        # keep optimising.
        while (iters_without_improvement < max_patience
               and iterations < max_iterations):
            iteration_time = time.perf_counter()
            # Compute the parameters and their cost after one more
            # optimisation iteration.
            params = optimiser.step(cost_function, params)
            cost = cost_function(params)

            if return_training_progress:
                # Save the cost of the newly computed parameters.
                iters_log["Gradient Descent Cost"].append(cost)

            if opt_cost - cost > step_eps:
                # The new parameters are better than the previous optimum.
                # Save the parameters.
                opt_params = params
                # Save their cost.
                opt_cost = cost
                # Reset the counter for the iterations without improvement.
                iters_without_improvement = 0

            else:
                # The new parameters are not better than the previous optimum.
                # Increase the counter for the iterations without improvement.
                iters_without_improvement = iters_without_improvement + 1

            if (iters_without_improvement > 0
               and iters_without_improvement % step_reduction_patience == 0):
                # Reduce the step size of the optimiser.
                new_opt_step_size = optimiser.stepsize * step_reduction_factor
                optimiser.stepsize = new_opt_step_size
                params = opt_params

                if verbose:
                    print(f"Step size reduced to {optimiser.stepsize} after"
                          f" {iters_without_improvement} iterations without"
                          " improvement.")

            iter_running_time = (time.perf_counter() - iteration_time)

            if return_training_progress:
                # Save the optimal cost until now.
                iters_log["Optimal Cost"].append(opt_cost)
                # Save the running time of this iteration.
                iters_log["Iteration Running Time"].append(iter_running_time)

            if verbose:
                print(f"Iteration {iterations+1}: Current minimum expected"
                      f" value of the circuit {opt_cost}. Time taken"
                      f" {iter_running_time} seconds.")

            if optimiser.stepsize < min_step_size:
                if verbose:
                    print("Optimisation stopped after the optimiser step size"
                          " was reduced below the minimum step size.")
                break

            iterations = iterations + 1

        if verbose:
            if iters_without_improvement >= max_patience:
                print("Optimisation stopped after the maximum number of"
                      " iterations without improvement was reached.")

        if return_training_progress:
            for key in iters_log:
                iters_log[key] = npp.array(iters_log[key])

        return opt_params, opt_cost, iters_log

    def GetMostProbableQuantumStates(self,
                                     wires: qml.wires.Wires,
                                     solution_prob_dist,
                                     count: int | None) -> pd.DataFrame:
        """Compute the bit-strings for which the probability of observing them
        is the highest.

        Parameters
        ----------
        wires : pennylane.wires.Wires
            Wires (qubits) in the circuit which labels are used to index the
            quantum states with the highest probabilities of being observed.
        solution_prob_dist : numpy.ndarray
            Optimiser probability distribution of pure quantum states.
        count : int
            If None, the probability distribution of all solutions is returned.
            Else, count many most probable solutions are returned.

        Returns
        -------
        most_probable_states : pandas.DataFrame
            pandas DataFrame with the bit-string in vector form as columns
            sorted from most probable to less probable. The first element of
            each row corresponds to the state of the first wire (qubit label).
        """

        # If the number of quantum states to return is None, return the whole
        # probability distribution of all possible quantum states.
        if count is None:
            return solution_prob_dist

        assert count > 0, \
            ("The number of quantum states to return is not strictly greater"
             " than zero.")

        # Select solutions_count many quantum states with the highest
        # probability from solution_prob_dist.
        most_probable_states = npp.argpartition(solution_prob_dist, -count)
        most_probable_states = most_probable_states[-count:]
        # Sort the top solutions_count many quantum states by their
        # probability, such that the quantum state with the highest
        # probability appears first in the most_probable_states arry.
        index_sorting = npp.argsort(-solution_prob_dist[most_probable_states])
        most_probable_states = most_probable_states[index_sorting]
        # Get the number of assets.
        N = self.NumberOfAssetsInPortfolio
        # most_probable_states only contains the integers corresponding to the
        # bit-strings of the quantum states.
        # We transform the integers into NumPy arrays of length 2*N which
        # encode the integers in binary form.
        # Each row contains a solution, sorted from top to bottom from highest
        # probability to lowest probability.
        most_probable_states = ((most_probable_states[:, None]
                                 & (1 << npp.arange(2*N))[::-1]) > 0)
        most_probable_states = most_probable_states.astype(int)

        most_probable_states = pd.DataFrame(data=most_probable_states.T,
                                            index=list(wires))

        return most_probable_states

    def GetMostProbablePortfolios(
            self,
            wires: qml.wires.Wires,
            circuit_probabilities,
            circuit_parameters,
            count: int) -> pd.DataFrame:
        """Compute the discrete portfolio vectors z in {-1, 0, 1}^N from the
        solutions to the optimisation problem enconded in the quantum circuit
        represented in circuit_probabilities.

        Parameters
        ----------
        wires : pennylane.wires.Wires
            Wires (qubits) in the circuit which labels are used to index and
            compute the portfolio corresponding to the quantum states with the
            highest probabilities of being observed.
        circuit_probabilities : pennylane.node
            Function which takes the circuit parameters as input, and returns
            the probability distribution of all possible qubit state strings.
        circuit_parameters : numpy.ndarray
            Parameters with which the circuit will be evaluated.
        count : int
            Number of most probable discrete portfolio vectors to return.

        Returns
        -------
        most_probable_pfs : pandas.DataFrame
            pandas DataFrame with the discrete portfolio vectors as columns
            sorted from most probable to less probable, and indexed by asset
            ticker in the rows.
        """

        # Check that at least one portfolio has to be returned.
        assert count > 0, \
            ("The number of portfolios to return is less or equal than 0.")

        PortfolioAssets = self.GetAssets()
        assert PortfolioAssets is not None, \
            ("There are no assets in the portfolio, which are needed for the"
             " indexing and mapping of quantum states to portfolios.")

        # Get the quantum states probability distribution using the circuit
        # with the optimal parameters just computed.
        solution_prob_dist = circuit_probabilities(circuit_parameters)
        # Get most probable quantum states.
        # Each row in most_probable_solutions contains a quantum state.
        most_probable_states = self.GetMostProbableQuantumStates(
            wires=wires,
            solution_prob_dist=solution_prob_dist,
            count=count)

        most_probable_pfs = {}
        for asset in PortfolioAssets:
            most_probable_pfs[asset] = (
                most_probable_states.loc[asset+"_long"]
                - most_probable_states.loc[asset+"_short"]
                )

        most_probable_pfs = pd.DataFrame.from_dict(data=most_probable_pfs,
                                                   orient="index")

        return most_probable_pfs

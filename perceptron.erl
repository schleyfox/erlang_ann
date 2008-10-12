% The Perceptron Server
%   represents a perceptron in a neural network
-module(perceptron).
-behaviour(gen_server).

% API
-export([start_link/0, connect/2, pass/2]).

% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2,
    code_change/3]).

-record(state, {inputs=[], outputs=[], 
                stale_inputs=[], stale_outputs=[]}).
-record(connection, {node, weight=0.5, value=0}).

% API

% Starts a server
start_link() -> gen_server:start_link(?MODULE, [], []).

% connects A to B where A and B are PIDs of perceptrons
connect(A, B) ->
  gen_server:cast(A, {connect_to_output, B}),
  gen_server:cast(B, {connect_to_input, A}).

stimulate(Node, Requester, Value) ->
  gen_server:cast(Node, {stimulate, Requester, self(), Value}).

pass(Node, Value) ->
  gen_server:cast(Node, {pass, self(), Value}).

backpropagate(Node, Weight, Delta) ->
  gen_server:cast(Node, {learn, self(), Weight, Delta}).


% gen_server callbacks

init([]) ->
  {ok, #state{}}.

handle_call(_Request, _From, State) ->
  {reply, ok, State}.

handle_cast({learn, From, Weight, Delta}, 
  #state{outputs = Outputs, stale_outputs = Stale_Outputs, inputs = Inputs}
  = State) ->

  {New_Outputs, New_Stale_Outputs} = 
    update_outputs(Outputs, Stale_Outputs, From, Weight, Delta),
  case New_Stale_Outputs of
    [] -> backpropagate_inputs(Inputs, delta_value(New_Outputs)),
          New_Inputs = update_input_weights(Inputs, Delta),
          {noreply, State#state{inputs = New_Inputs, 
                      outputs = New_Outputs,
                      stale_outputs = perceptron_nodes(New_Outputs)}};
    S -> {noreply, State#state{outputs = New_Outputs,
                              stale_outputs = S}}
  end;
handle_cast({pass, Requester, Value}, #state{outputs=Outputs} = State) ->
  stimulate_outputs(Outputs, Requester, (sigmoid_fun())(Value)),
  {noreply, State};
handle_cast({stimulate, Requester, From, Value}, 
  #state{inputs=Inputs, outputs=Outputs, stale_inputs = Stale_Inputs} = 
    State) ->

  {New_Inputs, New_Stale_Inputs} = 
    update_inputs(Inputs, Stale_Inputs, From, Value),
  Final_Stale_Inputs = case New_Stale_Inputs of
    [] -> stimulate_outputs(Outputs, Requester, output_value(New_Inputs)),
          perceptron_nodes(New_Inputs);
    List -> List
  end,
  {noreply, State#state{inputs=New_Inputs, 
      stale_inputs=Final_Stale_Inputs}};
handle_cast({connect_to_input, PID}, #state{inputs=Inputs} = State) ->
  {noreply,
    State#state{inputs = [#connection{node=PID}|Inputs]}};
handle_cast({connect_to_output, PID}, #state{outputs=Outputs} = State) ->
  {noreply,
    State#state{outputs = [#connection{node=PID}|Outputs]}};
handle_cast(_Msg, State) ->
  {noreply, State}.

handle_info(_Info, State) ->
  {noreply, State}.

terminate(_Reason, _State) ->
  ok.

code_change(_OldVsn, State, _Extra) ->
  {ok, State}.

% Internal Functions

% update the value of the input connection referenced by the node's PID
% in the list of inputs, and update the stale inputs list
update_inputs(Inputs, Stale_Inputs, Node, New_Value) ->
  {lists:map(
    fun(#connection{node=N} = C) ->
      case N of
        Node -> C#connection{value=New_Value};
        _ -> C
      end
    end,
    Inputs),
  lists:delete(Node, Stale_Inputs)}.

% updates the weight and delta of the output connection referenced by the
% PID node.  Updates stale outputs.
update_outputs(Outputs, Stale_Outputs, Node, Weight, New_Delta) ->
  {lists:map(
    fun(#connection{node=N} = C) ->
      case N of
        Node -> C#connection{weight=Weight, value=New_Delta};
        _ -> C
      end
    end,
    Outputs),
  lists:delete(Node, Stale_Outputs)}.

update_input_weights([], _Delta) -> [];
update_input_weights(Inputs, Delta) ->
  Output_Deriv = output_derivative_value(Inputs),
  Coefficient = learning_rate()*Delta*Output_Deriv,
  lists:map(
    fun(#connection{weight = W, value = V} = C) ->
      C#connection{weight = (W + (Coefficient*V))}
    end,
    Inputs).

learning_rate() -> 0.5.


% Sends the output value to the Requester
stimulate_outputs([], Requester, Output_Value) ->
  Requester ! {perceptron_output, self(), Output_Value};
% passes on the output value to the output connections
stimulate_outputs(Outputs, Requester, Output_Value) ->
  lists:foreach(
    fun(PID) ->
      stimulate(PID, Requester, Output_Value)
    end,
    perceptron_nodes(Outputs)).

% Sends the weight and delta value to connected inputs
backpropagate_inputs(Inputs, Delta_Value) ->
  lists:foreach(
    fun(#connection{node=PID, weight=Weight}) ->
      backpropagate(PID, Weight, Delta_Value)
    end,
    Inputs).

% returns a list of node PIDs from a list of connections
perceptron_nodes(Connections) ->
  lists:map(
    fun(#connection{node=N}) -> N end,
    Connections).

% returns a weight vector from a list of connections
weights(Connections) ->
  lists:map(
    fun(#connection{weight=W}) -> W end,
    Connections).

% returns a value vector from a list of connections
values(Connections) ->
  lists:map(
    fun(#connection{value=V}) -> V end,
    Connections).

% calculates the output value of the perceptron based on the inputs
% Uses the default sigmoid function as defined by sigmoid_fun/0
output_value(Inputs) ->
  output_value(sigmoid_fun(), Inputs).

% calculates the output value of the perceptron based on the inputs
% Uses a user specified sigmoid function
output_value(Func, Inputs) ->
  Func(dot_prod(weights(Inputs), values(Inputs))).

% calculates the output value using the sigmoid derivative function
output_derivative_value(Inputs) ->
  output_value(sigmoid_derivative_fun(), Inputs).

% calculates the delta value for the perceptron from the output list
delta_value(Outputs) ->
  dot_prod(weights(Outputs), values(Outputs)).

% Specifies the default sigmoid function
sigmoid_fun() ->
  fun(X) -> 1/(1+math:exp(-X)) end.

% Derivative of default sigmoid
sigmoid_derivative_fun() ->
  fun(X) -> math:exp(-X)/(1+math:exp(-2*X)) end.

% takes the dot product of two vectors, X and Y.
dot_prod(X, Y) -> dot_prod(0, X, Y).

dot_prod(Acc, [], []) -> Acc;
dot_prod(Acc, [X1|X], [Y1|Y]) ->
  dot_prod(X1*Y1 + Acc, X, Y).

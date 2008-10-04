% The Perceptron Server
%   represents a perceptron in a neural network
-module(perceptron).
-behaviour(gen_server).

% API
-export([start_link/0, connect/2]).

% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2,
    code_change/3]).

-record(state, {inputs=[], outputs=[]}).
-record(connection, {node, weight=0.5}).

% API

% Starts a server
start_link() -> gen_server:start_link(?MODULE, [], []).

% connects A to B where A and B are PIDs of perceptrons
connect(A, B) ->
  gen_server:cast(A, {connect_to_output, B}),
  gen_server:cast(B, {connect_to_input, A}).


% gen_server callbacks

init([]) ->
  {ok, #state{}}.

handle_call(_Request, _From, State) ->
  {reply, ok, State}.


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

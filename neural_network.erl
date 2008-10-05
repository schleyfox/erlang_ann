% Neural Network Server
%   Acts as the gateway to a cluster of perceptrons
-module(neural_network).
-behaviour(gen_server).

-export([start_link/0, calculate/1]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-record(state, {start_node}).

% API

start_link() ->
  gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

calculate(Value) ->
  gen_server:call(?MODULE, {calculate, Value}, infinity).

% gen_server callbacks

init([]) ->
  {ok, A} = perceptron:start_link(),
  {ok, B} = perceptron:start_link(),
  perceptron:connect(A,B),
  {ok, #state{start_node=A}}.

handle_call({calculate, Value}, _From, #state{start_node=Start} = State) ->
  perceptron:pass(Start, Value),
  receive
    {perceptron_output, _, Output_Value} -> 
      {reply, Output_Value, State}
  end;
handle_call(_Request, _From, State) ->
  Reply = ok,
  {reply, Reply, State}.

handle_cast(_Msg, State) ->
  {noreply, State}.

handle_info(_Info, State) ->
  {noreply, State}.

terminate(_Reason, _State) ->
  ok.

code_change(_OldVsn, State, _Extra) ->
  {ok, State}.

% Internal Functions

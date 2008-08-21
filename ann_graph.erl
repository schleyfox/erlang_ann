-module(ann_graph).
-export([grapher/2, output/2, start/0]).

start() -> 
  register(ann_grapher, spawn(fun() -> grapher(0, []) end)).

grapher(Frame, Connections) ->
  receive
    {connect, From, To} ->
      New_connections = [{connection, From, To}|Connections],
      output(Frame, New_connections),
      grapher(Frame + 1, New_connections);
    {stimulate, From, To, Value} ->
      Stim_connections = 
        stimulate_connection(From, To, Value, Connections),
      output(Frame, Stim_connections),
      grapher(Frame + 1, Connections);
    {output, From, Value} ->
      Output_connections = [{output, From, Value}|Connections],
      output(Frame, Output_connections),
      grapher(Frame + 1, Connections);
    {backprop, From, To, Value} ->
      Learning_connections = [{backpropagation, From, To, Value}|Connections],
      output(Frame, Learning_connections),
      grapher(Frame + 1, Connections)
      
  end.

output(Frame, Connections) ->
  file:write_file(sprintf("graph/~3.10.0B.dot", [Frame]),
    lists:concat([
      sprintf("digraph ann~w~n{~n", [Frame]) |
        lists:map(fun(Conn) ->
          case Conn of
          {connection, From, To} -> 
            sprintf("  \"~w\" -> \"~w\";~n", [From, To]);
          {stimulation, From, To, Value} ->
            sprintf("  \"~w\" -> \"~w\" [label=\"~w\" color=red];~n",
              [From, To, Value]);
          {output, From, Value} ->
            sprintf("  \"~w\" -> \"Output\" [label=\"~w\" color=green];~n\"Output\" [shape=box color=green]", [From, Value]);
          {backpropagation, From, To, Value} ->
            sprintf("  \"~w\" -> \"~w\" [label=\"~w\" color=blue];~n",
            [From, To, Value])

          end
          end,
          lists:reverse(Connections))] ++ 
      ["}"])).

stimulate_connection(From, To, Value, Connections) ->
  replace_elem({connection, From, To}, 
               {stimulation, From, To, Value},
               Connections).

replace_elem(Old, New, List) ->
  lists:map(fun(Elem) ->
      case Elem of
        Old -> New;
        _ -> Elem
      end
    end, List).

sprintf(Format, Args) ->
  io_lib:format(Format, Args).

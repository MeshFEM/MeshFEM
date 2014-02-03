
<%def name="header()">
library("ggplot2")

raw_data <- read.csv("${csv_file}");
names(raw_data) <- sub(" ", "_", names(raw_data));
summary(raw_data$Error);

raw_data <- subset(raw_data, Error < ${err_bound});
cat(nrow(raw_data), "samples within error bound.\n")
if (nrow(raw_data) == 0) {
    q();
}
</%def>

<%def name="filter_data()">
raw_data <- subset(raw_data, ${field_name} < ${upper_bound} & ${field_name} > ${lower_bound});
cat(nrow(raw_data), "samples left after filtering", "${field_name}", "\n")
if (nrow(raw_data) == 0) {
    print("No data left after filtering");
    q();
}
</%def>

<%def name="point_plot()">
p <- ggplot(raw_data);
p <- p + geom_point(aes(x=${x_col}, y=${y_col}, color=${w_col}));
p <- p + scale_color_gradient(low="blue", high="red");
% if title is not None:
p <- p + ggtitle("${title}");
% endif
</%def>

<%def name="histogram()">
p <- ggplot(raw_data);
p <- p + geom_histogram(aes(x=${w_col}), binwidth=0.001);
% if title is not None:
p <- p + ggtitle("${title}");
% endif
</%def>


<%def name="set_range()">
p <- p + xlim(${x_col_min}, ${x_col_max});
p <- p + ylim(${y_col_min}, ${y_col_max});
</%def>

<%def name="add_facet()">
p <- p + facet_grid(${facet_1} ~ ${facet_2});
</%def>

<%def name="save_plot()">
ggsave("${out_name}", width=${width}, height=${height});
</%def>

<%def name="add_points()">
<%
x_array = ",".join([str(entry) for entry in x]);
y_array = ",".join([str(entry) for entry in y]);
%>
x <- c(${x_array});
y <- c(${y_array});
p <- p + geom_point(aes(x=x, y=y, size=3), data = data.frame(x,y), color="green");
</%def>


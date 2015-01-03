
<%def name="header()">
library("ggplot2")

raw_data <- read.csv("${csv_file}");
names(raw_data) <- sub(" ", "_", names(raw_data));
summary(raw_data$Error);

</%def>

<%def name="filter_data()">
raw_data <- subset(raw_data, ${field_name} < ${upper_bound} & ${field_name} > ${lower_bound});
cat(nrow(raw_data), "samples left after filtering", "${field_name}", "\n")
if (nrow(raw_data) == 0) {
    print("No data left after filtering");
    q();
}
</%def>

<%def name="add_column()">
raw_data$${new_column} = with(raw_data, ${formula});
</%def>

<%def name="discretize_column()">
raw_data$${field_name} = as.factor(raw_data$${field_name});
</%def>

<%def name="scatter_plot()">
p <- ggplot(raw_data);
p <- p + geom_point(aes(x=${x_col}, y=${y_col}, color=${w_col}));
% if discrete:
#p <- p + scale_color_discrete();
p <- p + scale_colour_brewer(palette="Paired")
% else:
#p <- p + scale_color_gradient(low="blue", high="red");
p <- p + scale_color_gradientn(colours=c("blue", "green", "orange", "red"));
% endif
</%def>

<%def name="title()">
p <- p + ggtitle("${title_text}");
</%def>

<%def name="histogram()">
bin_width <- (max(raw_data$${w_col}) - min(raw_data$${w_col}))/${num_bins};
p <- ggplot(raw_data);
p <- p + geom_histogram(aes(x=${w_col}), binwidth=bin_width);
</%def>


<%def name="set_range()">
p <- p + xlim(${x_col_min}, ${x_col_max});
p <- p + ylim(${y_col_min}, ${y_col_max});
</%def>

<%def name="facet_grid()">
p <- p + facet_grid(${facet_1} ~ ${facet_2});
</%def>

<%def name="save_plot()">
ggsave("${out_name}", width=${width}, height=${height});
</%def>

<%def name="draw_ave_points()">
x <- mean(raw_data$${x_col});
y <- mean(raw_data$${y_col});
p <- p + annotate("text", x=x, y=y, label="${label}", hjust=0, vjust=1);
p <- p + annotate("point", x=x, y=y, color="green", size=3);
</%def>

<%def name="draw_point()">
p <- p + annotate("point", x=${x}, y=${y}, color="${color}", size=${size});
</%def>

<%def name="draw_text()">
p <- p + annotate("text", x=${x}, y=${y}, label="${text}", hjust=${hjust}, vjust=${vjust});
</%def>


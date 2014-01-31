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

<%def name="point_plot()">
p <- ggplot(raw_data);
p <- p + geom_point(aes(x=${x_col}, y=${y_col}, color=${w_col}));
p <- p + scale_color_gradient(low="blue", high="red");

<% with_facets = facet_1 is not None and facet_2 is not None %>
% if with_facets:
p <- p + facet_grid(${facet_1} ~ ${facet_2});
% endif

% if title is not None:
p <- p + ggtitle("${title}");
% endif

% if with_facets:
ggsave("${out_name}", width=20, height=12);
% else:
ggsave("${out_name}", width=10, height=6);
% endif
</%def>


<%def name="histogram()">
p <- ggplot(raw_data);
p <- p + geom_histogram(aes(x=${w_col}), binwidth=0.001);

<% with_facets = facet_1 is not None and facet_2 is not None %>
% if with_facets:
p <- p + facet_grid(${facet_1} ~ ${facet_2});
% endif

% if title is not None:
p <- p + ggtitle("${title}");
% endif

% if with_facets:
ggsave("${out_name}", width=20, height=12);
% else:
ggsave("${out_name}", width=10, height=6);
% endif
</%def>

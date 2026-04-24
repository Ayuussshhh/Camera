import UserLayout from "@/layouts/UserLayout";

export default function getUserLayout(config) {
  return function withUserLayout(page) {
    return <UserLayout {...config}>{page}</UserLayout>;
  };
}
